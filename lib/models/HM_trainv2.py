import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.utils.data as data
import numpy as np
from torch import nn
import time
from torch.optim.lr_scheduler import StepLR
from geomloss import SamplesLoss
import json

from lib.opt import parse_exp_args
from lib.utils.config import get_config
from lib.datasets import create_dataset
from ..utils.builder import MODEL
from time import time

import copy
class FreiHand_train(data.Dataset):
    def __init__(self):
        super(FreiHand_train, self).__init__()
        with open('/root/shenzhen11/yj_male/FreiHAND/training_xyz.json') as a:
            self.pose=np.array(json.load(a))
        with open('/root/shenzhen11/yj_male/FreiHAND/training_verts.json') as b:
            self.verts=np.array(json.load(b))
        self.diag_vertex = torch.load('Diag_Vertex.pth')
        self.L=5
        self.Pi=torch.acos(torch.zeros(1)).item() * 2


    def __getitem__(self, idx):

        return self.get_training_sample(idx)

    def get_training_sample(self, idx):
        """Get a FreiHAND sample for training
        """
        joint_cams=self.pose[idx,:,:]
        verts=self.verts[idx,:,:]
        root = joint_cams[0].copy()
        joint_cams -= root
        verts -= root
        joint_cams /= 0.2
        verts /= 0.2
        joint_cams=torch.from_numpy(joint_cams).float()
        verts=torch.from_numpy(verts).float()
        joint_cams=joints_cam_to_pose(joint_cams)
        gloinfo = joint_cams[:, 3:6].flatten()
        original_joint = joint_cams
        joint_cams=self.Position_encoder(joint_cams)
        verts = torch.mm(self.diag_vertex, verts)
        return joint_cams,verts,gloinfo,original_joint

    def __len__(self):
        return self.pose.shape[0]

    def Position_encoder(self,joint_cam):
        pos_code=torch.zeros(20,140)
        Pi=torch.acos(torch.zeros(1)).item() * 2
        for i in range(0,26):
            sita = joint_cam[:,i]
            if i<=5:
                #sita = joint_cam[i]
                for l in range(0,5):
                    pos_code[:,i * 10 + 2 * l]=torch.sin(pow(2,l)*Pi*sita)

                    pos_code[:,i * 10 + 2 * l+1] = torch.cos(pow(2, l) * Pi * sita)
            else:
                for l in range(0,2):
                    #print(60+(i-6)*3+2*l)
                    pos_code[:,60 + (i - 6) * 4 + 2 * l]=torch.sin(pow(2,l)*Pi*sita)
                    pos_code[:,60 + (i - 6) * 4 + 2 * l+1] = torch.cos(pow(2, l) * Pi * sita)
        return pos_code

def Position_encoder(joint_cam):
    batch_size,_,_=joint_cam.shape
    pos_code=torch.zeros(batch_size,20,140)
    # pos_code=torch.zeros(20,140)
    Pi=torch.acos(torch.zeros(1)).item() * 2
    for i in range(0,26):
        sita = joint_cam[:,:,i]
        # sita = joint_cam[:,i]
        if i<=5:
                #sita = joint_cam[i]
            for l in range(0,5):
                pos_code[:,:,i * 10 + 2 * l]=torch.sin(pow(2,l)*Pi*sita)

                pos_code[:,:,i * 10 + 2 * l+1] = torch.cos(pow(2, l) * Pi * sita)
        else:
            for l in range(0,2):
                #print(60+(i-6)*3+2*l)
                pos_code[:,:,60 + (i - 6) * 4 + 2 * l]=torch.sin(pow(2,l)*Pi*sita)
                pos_code[:,:,60 + (i - 6) * 4 + 2 * l+1] = torch.cos(pow(2, l) * Pi * sita)
    return pos_code

def edge_onehot_code(idx,num):
    onehot=torch.zeros([1,num],dtype=torch.float)
    onehot[:,idx]=1.
    return onehot

def joints_cam_to_pose(joint_cams):
    adj=np.array([[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],# W
                  [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T0
                  [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T1
                  [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T2
                  [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T3
                  [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#I0
                  [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#I1
                  [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],#I2
                  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#I3
                  [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],#M0
                  [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],#M1
                  [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],#M2
                  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],#M3
                  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],#R0
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],#R1
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],#R2
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],#R3
                  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],#L0
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],#L1
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],#L2
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])#L3
    
    edges_list = []
    for o in range(joint_cams.size(0)):
        order = 0
        for i in range(0, joint_cams.size(1)):
            for j in range(i, joint_cams.size(1)):
                if adj[i][j] == 1:
                    onehot = edge_onehot_code(order, 20).cuda()
                    edge_code=torch.cat((torch.unsqueeze(joint_cams[o][i,:],dim=0),torch.unsqueeze(joint_cams[o][j,:],dim=0),onehot),1)
                    if order==0:
                        edge=edge_code
                    else:
                        pre_edge=edge
                        edge=torch.cat((pre_edge,edge_code),0)
                    order += 1
        edges_list.append(edge)
    edges = torch.stack(edges_list, dim=0)
    return edges

def joints_cam_to_pose_for_feature(joint_cams):
    adj=np.array([[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],# W
                  [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T0
                  [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T1
                  [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T2
                  [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T3
                  [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#I0
                  [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#I1
                  [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],#I2
                  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#I3
                  [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],#M0
                  [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],#M1
                  [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],#M2
                  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],#M3
                  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],#R0
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],#R1
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],#R2
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],#R3
                  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],#L0
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],#L1
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],#L2
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])#L3
    
    edges_list = []
    for o in range(joint_cams.size(0)):
        order = 0
        for i in range(0, joint_cams.size(1)):
            for j in range(i, joint_cams.size(1)):
                if adj[i][j] == 1:
                    # onehot = edge_onehot_code(order, 20).cuda()
                    edge_code=torch.cat((torch.unsqueeze(joint_cams[o][i,:],dim=0),torch.unsqueeze(joint_cams[o][j,:],dim=0)),1)
                    if order==0:
                        edge=edge_code
                    else:
                        pre_edge=edge
                        edge=torch.cat((pre_edge,edge_code),0)
                    order += 1
        edges_list.append(edge)
    edges = torch.stack(edges_list, dim=0)
    return edges

class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*3, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU()
        )

    def forward(self,joints):
        y=self.linear_relu_stack(joints)
        return y

class ProjectionNet(nn.Module):
    def __init__(self):
        super(ProjectionNet, self).__init__()
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(140+100, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            )
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 100),
            # nn.LeakyReLU(),
        )

    def forward(self, x,z):
        x = torch.cat((x, z), 1)
        y = self.linear_relu_stack1(x)
        y=self.linear_relu_stack2(y)

        return y

class ProjectionNetv2(nn.Module):
    def __init__(self,n_view=1,view_channel=128):
        super(ProjectionNetv2, self).__init__()
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(140+100, 256),
            # nn.LeakyReLU(),
            )
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.linear_relu_stack3 = nn.Sequential(
            nn.Linear(256, 100),
        )
        self.n_view=n_view
        self.view_channel=view_channel

    def second_init(self):
        self.linear_relu_stack1_copy = copy.deepcopy(self.linear_relu_stack1)
        self.linear_relu_stack2_copy = copy.deepcopy(self.linear_relu_stack2)
        self.linear_relu_stack3_copy = copy.deepcopy(self.linear_relu_stack3)
        for p in self.linear_relu_stack1.parameters():
            p.requires_grad = False
        self.linear_relu_stack1.eval()
        for p in self.linear_relu_stack2.parameters():
            p.requires_grad = False
        self.linear_relu_stack2.eval()
        for p in self.linear_relu_stack3.parameters():
            p.requires_grad = False
        for p in self.linear_relu_stack1_copy.parameters():
            p.requires_grad = True
        self.linear_relu_stack1.eval()
        for p in self.linear_relu_stack2_copy.parameters():
            p.requires_grad = True
        self.linear_relu_stack2.eval()
        for p in self.linear_relu_stack3_copy.parameters():
            p.requires_grad = True
        self.linear_relu_stack3.eval()
        self.zero_convs=nn.Sequential(
            nn.Linear(2*self.n_view*self.view_channel, 140+100),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(100, 100),
        )
        # zero init
        for layer in self.zero_convs:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        return
    
    def forward(self, x,z):
        x = torch.cat((x, z), 1)
        y1 = self.linear_relu_stack1(x)
        y2=self.linear_relu_stack2(y1)
        y3=self.linear_relu_stack3(y2)
        return y3
    
    def forward_controlnet_branch1(self, x,z,c):
        c=self.zero_convs[0](c)
        x = torch.cat((x, z), -1)+c
        y1 = self.linear_relu_stack1_copy(x)
        y1 = self.zero_convs[1](y1)
        y2=self.linear_relu_stack2_copy(y1)
        y2 = self.zero_convs[2](y2)
        y3=self.linear_relu_stack3_copy(y2)
        y3=self.zero_convs[3](y3)
        return y3,y2,y1
    
    def forward_controlnet_branch2(self,x,z,y3_zeros_conved,y2_zeros_conved,y1_zeros_conved):
        x = torch.cat((x, z), -1)
        y1 = self.linear_relu_stack1(x)
        y1=y1+y1_zeros_conved
        y2=self.linear_relu_stack2(y1)
        y2=y2+y2_zeros_conved
        y3=self.linear_relu_stack3(y2)
        y3=y3+y3_zeros_conved
        return y3
    
    def forward_controlnet(self,x,z,c):
        y3_,y2_,y1_=self.forward_controlnet_branch1(x,z,c)
        y=self.forward_controlnet_branch2(x,z,y3_,y2_,y1_)

        return y
    

class HP2M(nn.Module):
    def __init__(self):
        super(HP2M, self).__init__()
        self.projection_x = ProjectionNet()
        self.projection_y = ProjectionNet()
        self.projection_z = ProjectionNet()
        self.globalnet = GlobalNet()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self,x,gloinfo):
        vertexnum = [45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62]
        batch,num,_=x.shape
        y=x.mT
        z = self.globalnet(gloinfo)
        #y=Position_encoder(x).mT.to(self.device)
        for i in range(0,num):
            #torch.cuda.synchronize()
            #time_start = time.time()
            #point=0.5*(x[:,0:3,i]+ x[:,3:6,i]).unsqueeze(1)
            #point=point.repeat(1,100,1)
            project_x = self.projection_x(y[:,:,i],z).unsqueeze(2)
            project_y = self.projection_y(y[:,:,i],z).unsqueeze(2)
            project_z = self.projection_z(y[:,:,i],z).unsqueeze(2)
            project=torch.stack([project_x,project_y,project_z],2)
            #print(project.shape)
            if i==0:
                projects=project[:,0:vertexnum[i]]
                #points=point
            else:
                projects=torch.cat([projects,project[:,0:vertexnum[i]]],1)
                #points = torch.cat([points, point], 1)
            #time_end = time.time()
            #time_sum = time_end - time_start
            #print(time_sum,1/time_sum)


        return projects.squeeze()#+points

class HP2Mv2(nn.Module):
    def __init__(self):
        super(HP2Mv2, self).__init__()
        self.projection_x = ProjectionNet()
        self.projection_y = ProjectionNet()
        self.projection_z = ProjectionNet()
        self.globalnet = GlobalNet()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self,x,gloinfo,original_x):
        vertexnum = [45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62]
        batch,num,_=x.shape
        y = x.permute(0, 2, 1)
        z = self.globalnet(gloinfo)
        original_x = original_x.permute(0, 2, 1)
        #y=Position_encoder(x).mT.to(self.device)
        for i in range(0,num):
            #torch.cuda.synchronize()
            #time_start = time.time()
            point = 0.5 * (original_x[:, 0:3, i] + original_x[:, 3:6, i]).unsqueeze(1)
            point = point.repeat(1, vertexnum[i], 1)
            project_x = self.projection_x(y[:,:,i],z).unsqueeze(2)
            project_y = self.projection_y(y[:,:,i],z).unsqueeze(2)
            project_z = self.projection_z(y[:,:,i],z).unsqueeze(2)
            project=torch.stack([project_x,project_y,project_z],2)
            #print(project.shape)
            if i==0:
                projects=project[:,0:vertexnum[i]]
                points=point
            else:
                projects=torch.cat([projects,project[:,0:vertexnum[i]]],1)
                points = torch.cat([points, point], 1)
            #time_end = time.time()
            #time_sum = time_end - time_start
            #print(time_sum,1/time_sum)


        return projects.squeeze()+points
    
    def forward_parallel(self,x,gloinfo,original_x):
        vertexnum = [45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62]
        batch,num,_=x.shape
        y = x.permute(0, 2, 1)
        z = self.globalnet(gloinfo)
        original_x = original_x.permute(0, 2, 1)
        #y=Position_encoder(x).mT.to(self.device)
        original_x = original_x.permute(0, 2, 1)
        for i in range(0,num):
            point = 0.5 * (original_x[:, 0:3, i] + original_x[:, 3:6, i]).unsqueeze(1)
            point = point.repeat(1, vertexnum[i], 1)
            if i==0:
                points=point
            else:
                points = torch.cat([points, point], 1)
                
        for i in range(0,num):
            project_x = self.projection_x(y[:,:,i],z).unsqueeze(2)
            project_y = self.projection_y(y[:,:,i],z).unsqueeze(2)
            project_z = self.projection_z(y[:,:,i],z).unsqueeze(2)
            project=torch.stack([project_x,project_y,project_z],2)
            #print(project.shape)
            if i==0:
                projects=project[:,0:vertexnum[i]]
            else:
                projects=torch.cat([projects,project[:,0:vertexnum[i]]],1)




        return projects.squeeze()+points

class HP2Mv3(nn.Module):
    def __init__(self,n_view=1,view_channel=128):
        super(HP2Mv3, self).__init__()
        self.n_view=n_view
        self.view_channel=view_channel
        self.projection_x = ProjectionNetv2(n_view=self.n_view,view_channel=self.view_channel)
        self.projection_y = ProjectionNetv2(n_view=self.n_view,view_channel=self.view_channel)
        self.projection_z = ProjectionNetv2(n_view=self.n_view,view_channel=self.view_channel)
        self.globalnet = GlobalNet()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vertexnum = [45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62]
        index=[]
        for i in range(20):
            index+=[i*100+k for k in range(self.vertexnum[i])]
        self.index=torch.Tensor(index).long()

    def second_init(self):
        # self.projection_x_copy = copy.deepcopy(self.projection_x)
        # self.projection_y_copy = copy.deepcopy(self.projection_y)
        # self.projection_z_copy = copy.deepcopy(self.projection_z)
        # for p in self.projection_x.parameters():
        #     p.requires_grad = False
        # self.projection_x.eval()
        # for p in self.projection_y.parameters():
        #     p.requires_grad = False
        # self.projection_y.eval()
        # for p in self.projection_z.parameters():
        #     p.requires_grad = False
        # self.projection_z.eval()
        # self.zero_convolution1d_1=nn.Linear(self.n_view*self.view_channel, 140+100)
        # self.zero_convolution1d_2=nn.Linear(256, 256)
        # self.zero_convolution1d_3=nn.Linear(256, 256)
        # self.zero_convolution1d_4=nn.Linear(256, 100)
        self.projection_x.second_init()
        self.projection_y.second_init()
        self.projection_z.second_init()
        return


    def forward(self,x,gloinfo,original_x,bone_aligned_feature):
        vertexnum = [45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62]
        batch,num,_=x.shape
        y = x.permute(0, 2, 1)
        z = self.globalnet(gloinfo)
        original_x = original_x.permute(0, 2, 1)
        #y=Position_encoder(x).mT.to(self.device)
        for i in range(0,num):
            #torch.cuda.synchronize()
            #time_start = time.time()
            point = 0.5 * (original_x[:, 0:3, i] + original_x[:, 3:6, i]).unsqueeze(1)
            point = point.repeat(1, vertexnum[i], 1)
            project_x = self.projection_x.forward_controlnet(y[:,:,i],z,bone_aligned_feature[:,i,:]).unsqueeze(2)
            project_y = self.projection_y.forward_controlnet(y[:,:,i],z,bone_aligned_feature[:,i,:]).unsqueeze(2)
            project_z = self.projection_z.forward_controlnet(y[:,:,i],z,bone_aligned_feature[:,i,:]).unsqueeze(2)
            project=torch.stack([project_x,project_y,project_z],2)
            #print(project.shape)
            if i==0:
                projects=project[:,0:vertexnum[i]]
                points=point
            else:
                projects=torch.cat([projects,project[:,0:vertexnum[i]]],1)
                points = torch.cat([points, point], 1)
            #time_end = time.time()
            #time_sum = time_end - time_start
            #print(time_sum,1/time_sum)


        return projects.squeeze()+points
    
    def forward_parallel(self,x,gloinfo,original_x,bone_aligned_feature):
        # bone_aligned_feature=bone_aligned_feature.permute(0,2,1)
        # vertexnum = [45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62]
        batch,num,_=x.shape
        # y = x.permute(0, 2, 1)
        z = self.globalnet(gloinfo).unsqueeze(-1).repeat(1,1,20).permute(0, 2, 1)
        original_x = original_x.permute(0, 2, 1)
        for i in range(0,num):
            point = 0.5 * (original_x[:, 0:3, i] + original_x[:, 3:6, i]).unsqueeze(1)
            point = point.repeat(1, self.vertexnum[i], 1)
            if i==0:
                points=point
            else:
                points = torch.cat([points, point], 1)

        #y=Position_encoder(x).mT.to(self.device)
        # time0=time()
        project_x = self.projection_x.forward_controlnet(x,z,bone_aligned_feature).unsqueeze(-1)
        project_y = self.projection_y.forward_controlnet(x,z,bone_aligned_feature).unsqueeze(-1)
        project_z = self.projection_z.forward_controlnet(x,z,bone_aligned_feature).unsqueeze(-1)
        project=torch.stack([project_x,project_y,project_z],-1).view(batch,-1,3)
        projects=project[:,self.index,:]
        # time1=time()
        # min_time=(time1-time0)
        # print(f'mlphand , {1/min_time} fps')
        # for i in range(0,num):
        #     #torch.cuda.synchronize()
        #     #time_start = time.time()
        #     project_x = self.projection_x.forward_controlnet(y[:,:,i],z,bone_aligned_feature[:,i,:]).unsqueeze(2)
        #     project_y = self.projection_y.forward_controlnet(y[:,:,i],z,bone_aligned_feature[:,i,:]).unsqueeze(2)
        #     project_z = self.projection_z.forward_controlnet(y[:,:,i],z,bone_aligned_feature[:,i,:]).unsqueeze(2)
        #     project=torch.stack([project_x,project_y,project_z],2)
        #     #print(project.shape)
        #     if i==0:
        #         projects=project[:,0:vertexnum[i]]
        #     else:
        #         projects=torch.cat([projects,project[:,0:vertexnum[i]]],1)
        #     #time_end = time.time()
        #     #time_sum = time_end - time_start
        #     #print(time_sum,1/time_sum)


        return projects.squeeze()+points



@MODEL.register_module()
class MLPHand(nn.Module):
    def __init__(self,model_path='/data/yangjian/POEM/checkpoints/stage1DexYCB/model_299.pth'):
        super(MLPHand,self).__init__()
        self.model=HP2Mv2()
        checkpoint=torch.load(model_path,map_location='cuda')
        print('load MLPHand model from: ',model_path)
        self.model.load_state_dict(checkpoint['net'])
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.diag_vertex = torch.load('/data/yangjian/POEM/our_code/Diag_Vertex.pth')       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inv_diag_vertex=torch.pinverse(self.diag_vertex).to(self.device)
        
        # 邻接矩阵
        adj=np.array([[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],# W
                    [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T0
                    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T1
                    [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T2
                    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#T3
                    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#I0
                    [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#I1
                    [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],#I2
                    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#I3
                    [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],#M0
                    [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],#M1
                    [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],#M2
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],#M3
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],#R0
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],#R1
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],#R2
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],#R3
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],#L0
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],#L1
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],#L2
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])#L3
        
        # 只取上三角矩阵中的非零元素索引
        parent_indices, child_indices = np.where(np.triu(adj, k=1) == 1)
        
        # 创建选择矩阵M1和M2
        self.M1 = torch.eye(21, dtype=torch.float32)[parent_indices].to(self.device)  # Size (20x21)
        self.M2 = torch.eye(21, dtype=torch.float32)[child_indices].to(self.device)   # Size (20x21)

    def forward(self,batch_pose):
        '''input can be non-root-relative skeletons with shape of (B,21,3) '''
        Batch_size,_,_=batch_pose.shape
        root=batch_pose[:,0,:].unsqueeze(1)
        batch_pose=batch_pose-root
        joint_cams=self.joints_cam_to_pose(batch_pose)
        gloinfo = joint_cams[:, :, 3:6].reshape(Batch_size, -1)
        original_joint = joint_cams
        pose = Position_encoder(joint_cams).cuda()
        pred=self.model(pose,gloinfo,original_joint)
        inv_diag_vertex_expanded = self.inv_diag_vertex.unsqueeze(0).repeat(batch_pose.size(0), 1, 1)
        pred=torch.bmm(inv_diag_vertex_expanded,pred)
        return pred+root
    
    def joints_cam_to_pose(self,joint_cams):
        '''
        inputs : batch x hand skeleton Bx21x3(joint_cams)
        implement : given 2 Matrix M1 for parent and M2 for child with size (20x21), conduct matrix left multiplication 
                    with joint_cams obtain parent joints(20x3) and child joints(20x3), then concat them with onehot 
                    matrix(20x20) obtain final output
        output : batch x 20 x bone vector(in R^(3+3+20))
        ''' 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        joint_cams = joint_cams.to(device)

        M1 = self.M1.repeat(joint_cams.size(0), 1, 1)
        M2 = self.M2.repeat(joint_cams.size(0), 1, 1)

        # 获得父节点和子节点的坐标
        parent_joints = torch.matmul(M1, joint_cams)  # Size (batch x 20 x 21) @ (batch x 21 x 3) = (batch x 20 x 3)
        child_joints = torch.matmul(M2, joint_cams)   # Size (batch x 20 x 21) @ (batch x 21 x 3) = (batch x 20 x 3)
        
        # 创建骨骼类型的one-hot编码
        edge_types = torch.eye(20, dtype=torch.float32).to(device).unsqueeze(0).repeat(joint_cams.size(0), 1, 1)  # Size (batch x 20 x 20)

        # 将父关节坐标、子关节坐标和one-hot编码连接起来
        edges = torch.cat((parent_joints, child_joints, edge_types), dim=2)  # Size (batch x 20 x (3+3+20))
        return edges

@MODEL.register_module()
class MLPHand_v2(nn.Module):
    def __init__(self,model_path='/data/yangjian/POEM/checkpoints/stage1DexYCB/model_299.pth',n_view=4,view_channel=128):
        super(MLPHand_v2,self).__init__()
        self.n_view=n_view
        self.view_channel=view_channel
        self.model=HP2Mv3(n_view=self.n_view,view_channel=self.view_channel)
        checkpoint=torch.load(model_path,map_location='cuda')
        print('load MLPHand model from: ',model_path)
        s=checkpoint['net']
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('linear_relu_stack2.2', 'linear_relu_stack3.0')] = v
        self.model.load_state_dict(new_s,strict=True)
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.eval()
        self.model.second_init()
        print('second init done')

        self.diag_vertex = torch.load('/data/yangjian/POEM/our_code/Diag_Vertex.pth')       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inv_diag_vertex=torch.pinverse(self.diag_vertex).to(self.device)

        # 0-手腕, 1-4拇指, 5-8食指, 9-12中指, 13-16无名指, 17-20小指
        parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        children = [1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, -1, 13, 14, 15, -1, 17, 18, 19, -1]
        # 创建父矩阵M1和子矩阵M2
        self.M1 = torch.zeros(20, 21).cuda()
        self.M2 = torch.zeros(20, 21).cuda()
        for i, (p, c) in enumerate(zip(parents[1:], children)):
            self.M1[i, p] = 1
            if c != -1:
                self.M2[i, c] = 1
        self.one_hot = torch.stack([edge_onehot_code(i, 20) for i in range(20)]).squeeze(1).cuda()
        self.pos_code=torch.zeros(20,140).cuda()
        self.min_time=100
        self.min_time2=100


    def Position_encoder(self,joint_cam,pos_code):
        batch_size,_,_=joint_cam.shape
        # pos_code=torch.zeros(batch_size,20,140)
        # pos_code=torch.zeros(20,140)
        Pi=torch.acos(torch.zeros(1)).item() * 2
        for i in range(0,26):
            sita = joint_cam[:,:,i]
            # sita = joint_cam[:,i]
            if i<=5:
                    #sita = joint_cam[i]
                for l in range(0,5):
                    pos_code[:,:,i * 10 + 2 * l]=torch.sin(pow(2,l)*Pi*sita)

                    pos_code[:,:,i * 10 + 2 * l+1] = torch.cos(pow(2, l) * Pi * sita)
            else:
                for l in range(0,2):
                    #print(60+(i-6)*3+2*l)
                    pos_code[:,:,60 + (i - 6) * 4 + 2 * l]=torch.sin(pow(2,l)*Pi*sita)
                    pos_code[:,:,60 + (i - 6) * 4 + 2 * l+1] = torch.cos(pow(2, l) * Pi * sita)

        return pos_code    

    def forward(self,batch_pose):
        '''input can be non-root-relative skeletons with shape of (B,21,3) '''
        Batch_size,_,_=batch_pose.shape
        root=batch_pose[:,0,:].unsqueeze(1)
        batch_pose=batch_pose-root
        joint_cams=joints_cam_to_pose(batch_pose)
        gloinfo = joint_cams[:, :, 3:6].reshape(Batch_size, -1)
        original_joint = joint_cams
        pose = Position_encoder(joint_cams).cuda()
        pred=self.model(pose,gloinfo,original_joint)
        inv_diag_vertex_expanded = self.inv_diag_vertex.unsqueeze(0).repeat(batch_pose.size(0), 1, 1)
        pred=torch.bmm(inv_diag_vertex_expanded,pred)
        return pred+root

    def forward_controlnet(self,batch_pose,pose_aligned_feature):
        '''input can be non-root-relative skeletons with shape of (B,21,3) '''
        # time0=time()
        Batch_size,_,_=batch_pose.shape
        root=batch_pose[:,0,:].unsqueeze(1)
        batch_pose=batch_pose-root
        joint_cams=self.joints_cam_to_pose(batch_pose)
        joint_cams_feature=self.joints_cam_to_pose_for_feature(pose_aligned_feature)
        gloinfo = joint_cams[:, :, 3:6].reshape(Batch_size, -1)
        original_joint = joint_cams
        pose = self.Position_encoder(joint_cams,self.pos_code.repeat(joint_cams.size(0),1,1))
        # time1=time()
        # if self.min_time2>(time1-time0):
        #     self.min_time2=time1-time0
        #     print(f"order encode has {1/self.min_time} fps")
        # time0=time()
        pred=self.model.forward_parallel(pose,gloinfo,original_joint,joint_cams_feature)
        # time1=time()
        # if self.min_time>(time1-time0):
        #     self.min_time=time1-time0
        #     print(f"mlphand has {1/self.min_time} fps")
        inv_diag_vertex_expanded = self.inv_diag_vertex.unsqueeze(0).repeat(batch_pose.size(0), 1, 1)
        pred=torch.bmm(inv_diag_vertex_expanded,pred)
        return pred+root
    
    def joints_cam_to_pose(self,joint_cams):
        '''
        inputs : batch x hand skeleton Bx21x3(joint_cams)
        implement : given 2 Matrix M1 for parent and M2 for child with size (20x21), conduct matrix left multiplication 
                    with joint_cams obtain parent joints(20x3) and child joints(20x3), then concat them with onehot 
                    matrix(20x20) obtain final output
        output : batch x 20 x bone vector(in R^(3+3+20))
        ''' 
        # 计算父关节和子关节的位置
        parent_joints = torch.matmul(self.M1, joint_cams)
        child_joints = torch.matmul(self.M2, joint_cams)
        
        # 创建one-hot编码矩阵
        # one_hot = torch.stack([edge_onehot_code(i, 20) for i in range(20)]).squeeze(1)
        one_hot = self.one_hot.to(device=joint_cams.device, dtype=joint_cams.dtype).expand(joint_cams.size(0), -1, -1)
        
        # 拼接结果
        edges = torch.cat([parent_joints, child_joints, one_hot], dim=2)
        return edges
    
    def joints_cam_to_pose_for_feature(self,joint_cams):

        # 计算父关节和子关节的位置
        parent_joints = torch.matmul(self.M1, joint_cams)
        child_joints = torch.matmul(self.M2, joint_cams)
        
        # 拼接结果
        edges = torch.cat([parent_joints, child_joints], dim=2)
        return edges
    

class Vert_reg_loss(nn.Module):

    def __init__(self):
        super(Vert_reg_loss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Loss_fn= torch.nn.MSELoss(reduce=True, size_average=True)
        #self.part_vert_idx=torch.from_numpy(np.load('part_vert_idx_final.npy')).to(self.device)
        #self.part_face_idx=torch.from_numpy(np.load('part_face_idx_final_v2.npy')).to(self.device)


    def forward(self,predvert,gtvert):
        batch_size,_ ,_ = predvert.shape
        Loss=self.Loss_fn(predvert,gtvert)

        return Loss

class Part_wise_EMD(nn.Module):

    def __init__(self):
        super(Part_wise_EMD, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Emd=SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.vertexnum = torch.tensor([0,45, 61, 43, 45, 92, 34, 41, 62, 44, 44, 58, 42, 40, 60, 41, 35, 64, 28, 50, 62])
        #self.part_vert_idx=torch.from_numpy(np.load('part_vert_idx_final.npy')).to(self.device)
        #self.part_face_idx=torch.from_numpy(np.load('part_face_idx_final_v2.npy')).to(self.device)


    def forward(self,predvert,batch_pointclouds):
        batch_size,_ ,_ = predvert.shape
        Loss=torch.tensor(0.).to(self.device)

        #batch_pointclouds=1000*batch_pointclouds.to(self.device).float()
        #Loss=self.Emd(predvert,batch_pointclouds)

        for i in range(0,20):
            predpoint=predvert[:,self.vertexnum[0:i+1].sum():self.vertexnum[0:i+1].sum()+self.vertexnum[i+1],:]
            supervisepoint=batch_pointclouds[:,self.vertexnum[0:i+1].sum():self.vertexnum[0:i+1].sum()+self.vertexnum[i+1],:]
            loss1=self.Emd(predpoint.contiguous(),supervisepoint.contiguous())
            Loss+=loss1.sum()/batch_size

        return Loss

def multiview_batch_to_singleview_batch(multiview_batch):
    sample_shape=multiview_batch.shape[-2:]
    return multiview_batch.view(-1,sample_shape[0],sample_shape[1])


if __name__ == "__main__":
    """
    train the skeleton2mesh module
    """
    save_dir='/data/yangjian/POEM/checkpoints/stage1_mat'
    checkpoint_path='/data/yangjian/POEM/checkpoints/stage1/model_299.pth'
    pretrained_model=True
    #torch.multiprocessing.set_start_method('spawn')

    #parameters
    Trainepochs=300
    Batch_size=256
    save_epoch=10
    start_epoch=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dataset
    # Mydataset = FreiHand_train()
    # Mydataset = FreiHand_train()
    # data_loader=data.DataLoader(Mydataset,batch_size=Batch_size,shuffle=True,num_workers=16)

    sys.argv = ["HM_trainv2.py", "--cfg", "/data/yangjian/POEM/config/release/POEM_OakInkMV.yaml", "-g", "6,7", "-w", "16"]
    arg, _ = parse_exp_args()
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    arg.world_size = arg.n_gpus * arg.nodes
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    arg.batch_size=64
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size
    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)

    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    data_loader = data.DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)
    
    test_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    test_loader = data.DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)

    # model
    model = HP2Mv2()
    loss5=Vert_reg_loss()
    lamda5=10

    #optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    model.to(device)

    loss_record=[]
    if  pretrained_model:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        #optimizer.state_dict()['param_groups'][0]['lr']=5e-5

    diag_vertex = torch.load('/data/yangjian/POEM/code/Diag_Vertex.pth')
    inv_diag_vertex=torch.pinverse(diag_vertex)
    
    print('start trainning...')
    for epoch in range(start_epoch,Trainepochs):
        #(optimizer.state_dict()['param_groups'][0]['lr'])
        time0=time.time()
        global_loss=0
        global_surface_loss = 0
        global_sphere_loss = 0
        global_face2point_loss=0
        support_Loss=0
        reg_Loss=0
        #EMD_loss=0

        for step, batch in enumerate(data_loader):
            pose_rel=multiview_batch_to_singleview_batch(batch['target_joints_3d_rel'])
            verts_rel=multiview_batch_to_singleview_batch(batch['target_verts_3d_rel'])
            # pose_rel=batch['target_joints_3d_rel']
            # verts_rel=batch['target_verts_3d_rel']

            joint_cams=pose_rel.float()
            vert_batch=verts_rel.float()
            joint_cams=MLPHand.joints_cam_to_pose(joint_cams)
            gloinfo = joint_cams[:, :, 3:6].reshape(4*arg.batch_size, -1)
            original_joint = joint_cams
            pose = Position_encoder(joint_cams)

            pose=pose.to(device)
            verts = vert_batch.to(device)
            gloinfo=gloinfo.to(device)
            original_joint=original_joint.to(device)
            # 扩展diag_vertex的维度以匹配批量乘法的要求
            diag_vertex_expanded = diag_vertex.unsqueeze(0).repeat(vert_batch.size(0), 1, 1).to(device)
            # 使用torch.bmm进行批量矩阵乘法
            verts = torch.bmm(diag_vertex_expanded, verts)

            verts=1000*verts
            verts=torch.squeeze(verts)
            
            # 到这一步是[batch, 20, 26], gloinfo是被展平成[1248]
            # 接下来想一次处理一个batch，但是GlobalNet模型的初始化好像写死了处理[60], 就是[20, 26]展平
            pred=model(pose,gloinfo,original_joint)
            pred=1000*pred
            loss_5=loss5(pred,verts)
            loss=lamda5*loss_5

            global_loss+=loss
            reg_Loss+=loss_5

            optimizer.zero_grad()
            loss.backward()  # loss反向传播
            optimizer.step()
            print('epoch:'+str(epoch)+'  total loss : {:.5f},  vert loss : {:.5f}, this epoch take {:.5f}s  lr:{}'.\
              format(global_loss/(step+1),reg_Loss/(step+1),time.time()-time0,optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()  # 反向传播后参数更新
        

        loss_record.append('epoch:'+str(epoch)+'  total loss : {:.5f},  emd_Loss : {:.5f}, this epoch take {:.5f}s  lr:{}'.\
              format(global_loss/(step+1),reg_Loss/(step+1),time.time()-time0,optimizer.state_dict()['param_groups'][0]['lr']))

        if (epoch+1)%save_epoch==0:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, save_dir+'/model_{}.pth'.format(epoch))
            loss_record = np.array(loss_record)
            np.save(save_dir+'/loss_{}.npy'.format(epoch),loss_record)
            loss_record = loss_record.tolist()
    print('Training Done')
    print('Begin testing')
    with torch.no_grad():
        reg_Loss=0
        time0=time.time()
        for step, batch in enumerate(test_loader):
            pose_rel=multiview_batch_to_singleview_batch(batch['target_joints_3d_rel'])
            
            verts_rel=multiview_batch_to_singleview_batch(batch['target_verts_3d_rel'])
            # print(verts_rel.max())
            # pose_rel=batch['target_joints_3d_rel']
            # verts_rel=batch['target_verts_3d_rel']

            joint_cams=pose_rel.float()
            vert_batch=verts_rel.float()
            joint_cams=MLPHand.joints_cam_to_pose(joint_cams)
            gloinfo = joint_cams[:, :, 3:6].reshape(4*arg.batch_size, -1)
            original_joint = joint_cams
            pose = Position_encoder(joint_cams)
            pose=pose.to(device)
            verts = vert_batch.to(device)
            gloinfo=gloinfo.to(device)
            original_joint=original_joint.to(device)
            # 扩展diag_vertex的维度以匹配批量乘法的要求
            diag_vertex_expanded = diag_vertex.unsqueeze(0).repeat(vert_batch.size(0), 1, 1).to(device)
            # 使用torch.bmm进行批量矩阵乘法
            verts = torch.bmm(diag_vertex_expanded, verts)
            verts=1000*verts # convert to mm unit
            verts=torch.squeeze(verts)
            
            # 到这一步是[batch, 20, 26], gloinfo是被展平成[1248]
            # 接下来想一次处理一个batch，但是GlobalNet模型的初始化好像写死了处理[60], 就是[20, 26]展平
            pred=model(pose,gloinfo,original_joint)
            pred=1000*pred
            inv_diag_vertex_expanded = inv_diag_vertex.unsqueeze(0).repeat(vert_batch.size(0), 1, 1).to(device)
            loss_5=torch.norm(torch.bmm(inv_diag_vertex_expanded,pred-verts),p=2,dim=-1,keepdim=True)
            # loss=lamda5*loss_5

            # global_loss+=loss
            reg_Loss+=loss_5.mean()
            print('step:'+str(step)+'  total loss : {:.5f},  Vertex Error : {:.5f}, this epoch take {:.5f}s  lr:{}'.\
              format(0/(step+1),reg_Loss/(step+1),time.time()-time0,optimizer.state_dict()['param_groups'][0]['lr']))