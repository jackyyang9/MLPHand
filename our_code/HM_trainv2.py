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
                    onehot = edge_onehot_code(order, 20)
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
            # nn.LeakyReLU(),
            nn.Linear(256, 100),
            # nn.LeakyReLU(),
        )

    def forward(self, x,z):
        x = torch.cat((x, z), 1)
        y = self.linear_relu_stack1(x)
        y=self.linear_relu_stack2(y)

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

class MLPHand(nn.Module):
    def __init__(self,model_path='/data/lijiakun/POEM/checkpoints/stage1/model_299.pth'):
        super(MLPHand,self).__init__()
        self.model=HP2Mv2()
        checkpoint=torch.load(model_path)
        print('load MLPHand model from: ',model_path)
        self.model.load_state_dict(checkpoint['net'])
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.diag_vertex = torch.load('/data/lijiakun/POEM/code/Diag_Vertex.pth')       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inv_diag_vertex=torch.pinverse(self.diag_vertex).to(self.device)

    def forward(self,batch_pose):
        '''input can be non-root-relative skeletons with shape of (B,21,3) '''
        Batch_size,_,_=batch_pose.shape
        root=batch_pose[:,0,:]
        batch_pose=batch_pose-root
        joint_cams=joints_cam_to_pose(batch_pose)
        gloinfo = joint_cams[:, :, 3:6].reshape(Batch_size, -1)
        original_joint = joint_cams
        pose = Position_encoder(joint_cams)
        pred=self.model(pose,gloinfo,original_joint)
        inv_diag_vertex_expanded = self.inv_diag_vertex.unsqueeze(0).repeat(batch_pose.size(0), 1, 1)
        pred=torch.bmm(inv_diag_vertex_expanded,pred)
        return pred+root



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
    save_dir='/data/yangjian/POEM/checkpoints/Oaklnk_linear_modeling'
    checkpoint_path=''
    pretrained_model=True
    #torch.multiprocessing.set_start_method('spawn')

    #parameters
    Trainepochs=300
    Batch_size=256
    save_epoch=10
    start_epoch=0
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    #dataset
    # Mydataset = FreiHand_train()
    # Mydataset = FreiHand_train()
    # data_loader=data.DataLoader(Mydataset,batch_size=Batch_size,shuffle=True,num_workers=16)

    sys.argv = ["HM_trainv2.py", "--cfg", "/data/lijiakun/POEM/config/release/POEM_OakInkMV.yaml", "-g", "4", "-w", "16"]
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

    diag_vertex = torch.load('/data/lijiakun/POEM/code/Diag_Vertex.pth')
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
            joint_cams=joints_cam_to_pose(joint_cams)
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
            joint_cams=joints_cam_to_pose(joint_cams)
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