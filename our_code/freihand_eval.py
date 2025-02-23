import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.utils.data as data
import numpy as np
#import openmesh as om
from torch import nn
import torch.nn.functional as F
from pytorch3d.io import load_obj,save_obj,save_ply
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.loss import point_mesh_face_distance
import time
from torch.optim.lr_scheduler import StepLR
from  scipy.spatial import Delaunay
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from HM_trainv2 import Vert_reg_loss,Part_wise_EMD,HP2M,HP2Mv2
import json
from scipy.linalg import orthogonal_procrustes
from torchinfo import summary

class FreiHand_test(data.Dataset):
    def __init__(self):
        super(FreiHand_test, self).__init__()
        with open('/root/shenzhen11/yj_male/FreiHAND_eval/evaluation_xyz.json') as a:
            self.pose=np.array(json.load(a))
        with open('/root/shenzhen11/yj_male/FreiHAND_eval/evaluation_verts.json') as b:
            self.verts=np.array(json.load(b))
        with open('/root/shenzhen11/yj_male/meshgraphomer/pred.json') as f:
            self.meshgraphomer_out_list = json.load(f)

        self.pose_list = np.array(self.meshgraphomer_out_list[0])
        self.pose_list = np.array(self.pose)
        self.diag_vertex = torch.load('Diag_Vertex.pth')
        self.L=5
        self.Pi=torch.acos(torch.zeros(1)).item() * 2



    def __getitem__(self, idx):

        return self.get_training_sample(idx)

    def get_training_sample(self, idx):
        meshgraphomer_mesh_poses = self.pose_list[idx, :, :]
        root_mg = meshgraphomer_mesh_poses[0].copy()
        meshgraphomer_mesh_poses-=root_mg
        joint_cams=self.pose[idx, :, :]
        verts=self.verts[idx,:,:]
        root = joint_cams[0].copy()
        joint_cams -= root
        verts -= root
        #xyz-=root
        xyz = joint_cams.copy()
        noise = 0.001 * np.random.randn(21, 3) * np.sqrt(0)
        joint_cams += noise
        noise_xyz = joint_cams+noise
        joint_cams = self.pose[idx, :, :]
        verts = self.verts[idx, :, :]
        root = joint_cams[0].copy()
        joint_cams -= root
        # joint_cams = procruste_numpy(xyz,joint_cams)
        #xyz = joint_cams.copy()
        joint_cams /= 0.2
        verts /= 0.2
        meshgraphomer_mesh_poses=procruste_numpy(xyz,meshgraphomer_mesh_poses)
        meshgraphomer_mesh_poses /= 0.2
        meshgraphomer_mesh_poses = torch.from_numpy(meshgraphomer_mesh_poses).float()
        joint_cams=torch.from_numpy(joint_cams).float()
        verts=torch.from_numpy(verts).float()
        #noise=5*0.001*pow(0,0.5)*torch.randn(20,6)
        #joint_cams[:,0:6]=joint_cams[:,0:6]+noise

        joint_cams=joints_cam_to_pose(meshgraphomer_mesh_poses)
        gloinfo = joint_cams[:, 3:6].flatten()
        original_joint = joint_cams
        joint_cams=self.Position_encoder(joint_cams)
        #sample_dir=base_dir+'parts_500_poisson_global/'
        #sample_point=torch.load(sample_dir+'points_{}'.format(file_num))

        #print(joint_cams.shape)
        #verts = torch.mm(self.diag_vertex, verts)
        return joint_cams,verts,xyz,gloinfo,noise_xyz,original_joint

    def __len__(self):
        return self.pose.shape[0]

    def Position_encoder(self,joint_cam):
        pos_code=torch.zeros(20,140)
        Pi=self.Pi
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




def edge_onehot_code(idx,num):
    onehot=torch.zeros([1,num],dtype=torch.float)
    onehot[:,idx]=1.
    return onehot

def joints_cam_to_pose(joints_cam):

    order=0
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
    edge=[]
    for i in range(0,len(joints_cam)):
        for j in range(i,len(joints_cam)):
            if adj[i][j]==1:
                onehot=edge_onehot_code(order,20)
                edge_code=torch.cat((torch.unsqueeze(joints_cam[i,:],dim=0),torch.unsqueeze(joints_cam[j,:],dim=0),onehot),1)

                if order==0:
                    edge=edge_code
                else:
                    pre_edge=edge
                    edge=torch.cat((pre_edge,edge_code),0)
                order += 1

    return edge

def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2
    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2
    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t

def procruste_numpy(mesh,mesh_pred):
    mesh_pred=align_w_scale(mesh,mesh_pred)
    return mesh_pred

def procruste_eval(mesh,mesh_pred):
    device_g = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_c=torch.device("cpu")
    batch,_,_=mesh.shape
    for i in range(batch):
        mesh_i=mesh[i,:,:].to(device_c).numpy()
        mesh_pred_i=mesh_pred[i,:,:].to(device_c).numpy()
        mesh_pred_i=align_w_scale(mesh_i,mesh_pred_i)
        mesh_pred[i,:,:]=torch.from_numpy(mesh_pred_i).to(device_g)

    return mesh_pred
def MANO2MPII(xyz):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    M = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]).float().cuda()
    batch=xyz.shape[0]
    M=M.repeat(batch,1,1)
    return torch.bmm(M,xyz)

def Mean_vert_distance(batchA,batchB):
    Norm=torch.norm(batchA-batchB,p=2,dim=-1,keepdim=True)

    return Norm.mean()

if __name__ == "__main__":
    checkpoint_dir='/root/shenzhen11/yj_male/HandMesh/checkpoints/MeshHand/hidden512/model_199.pth'
    Mydataset = FreiHand_test()
    Trainepochs=100
    Batch_size=180
    save_epoch=100
    start_epoch=0
    data_loader=data.DataLoader(Mydataset,batch_size=Batch_size,shuffle=False,num_workers=16)
    model = HP2Mv2()
    # summary(model,((1,20,140),(1,60),(1,20,6)))
    # sys.exit()
    loss5 = Vert_reg_loss()
    loss6 = Vert_reg_loss()
    lamda1=1
    lamda2=1
    lamda3=1
    lamda4=1
    lamda5=1
    lamda6=1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loss_record=[]
    A=torch.load('Diag_Vertex.pth').T
    #A=torch.mm(A.T,A)
    B=torch.mm(torch.load("diag_scale_matrix.pth"),A)
    _, faces_index, _ = load_obj('210.obj')
    faces_index=faces_index.verts_idx
    j_reg=torch.from_numpy(np.load('j_reg.npy')).to(device)
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['net'])
    print('start testing...')
    time0=time.time()
    global_loss=0
    global_surface_loss = 0
    global_sphere_loss = 0
    global_face2point_loss=0
    #Std=torch.zeros(20)
    #Mean=torch.zeros(20)
    #support_Loss=0
    PV=0
    PJ=0
    J_error=0
    P_J_error = 0
    #count=torch.zeros(21)
    loss_count=torch.zeros([1,1]).to(device)
    with torch.no_grad():
        for step ,[pose , verts ,xyz,gloinfo,noise_xyz,original_joint] in enumerate(data_loader):
            print(step)
            #print(step)
            batch=pose.shape[0]
            pose=pose.to(device)
            verts = verts.to(device)
            gloinfo=gloinfo.to(device)
            original_joint=original_joint.to(device)
            xyz=1000*xyz.to(device)
            noise_xyz = 1000 * noise_xyz.to(device)
            #pose=pose
            verts=200*verts
            verts=torch.squeeze(verts)
            #torch.cuda.synchronize()
            #time_start = time.time()
            pred = model(pose,gloinfo,original_joint)
            #torch.cuda.synchronize()
            #time_end = time.time()
            #time_sum = time_end - time_start
            #print(time_sum,1/time_sum)


            pred=200*pred
            #points=200*points
            if Batch_size==1:
                verts=verts.unsqueeze(0)
                #faces=faces.unsqueeze(0)
            #     pred=pred.unsqueeze(0)
            #handmesh=Meshes(verts=verts, faces=faces)
            batchB=B.repeat(batch,1,1).to(device)
            J_reg=j_reg.repeat(batch,1,1).float()
            print(verts.shape)
            pred=torch.bmm(batchB,pred)
            pred=procruste_eval(verts,pred)
            pred_J = torch.bmm(J_reg, pred)
            pred_J = MANO2MPII(pred_J)
            pred_J = procruste_eval(xyz, pred_J)
            # if step==1:
            save_obj("./show/test{}.obj".format(step),pred[0,:,:],faces_index)
            save_obj("./show/testGT{}.obj".format(step), verts[0, :, :], faces_index)
                # save_ply("test_pose.ply", pred_J[0, :, :])

            #loss_5=loss5(pred,verts)
            loss_5=Mean_vert_distance(pred,verts)
            #loss_6 = loss6(pred_J, xyz)
            loss_6 = Mean_vert_distance(pred_J, xyz)
            loss_7 = Mean_vert_distance(noise_xyz, xyz)
            noise_xyz = procruste_eval(xyz, noise_xyz)
            loss_8 = Mean_vert_distance(noise_xyz, xyz)

            #loss_6,table=lamda6*loss6(handmesh,pred)
            #loss_count=torch.cat([loss_count,torch.tensor([[loss_5]]).to(device)],0)
            #loss=lamda5*loss_5#+loss_6#+loss_3#+loss_1#+loss_4
            PV+=loss_5
            PJ+=loss_6
            J_error+=loss_7
            P_J_error+=loss_8
            #std,mean=uniformity_metric(200*batchpoints)
            #batchsize,num,vector_length=batchpoints.shape
            #batchpoints+=(1/200)*pow(20,0.5)*torch.randn(batchsize,num,vector_length)
            #std, mean = uniformity_metric(pred)
            #Std+=std
            #Mean+=mean
            #stastic
            # table=torch.sqrt(table)
            # for ths in range(20,21):
            #     y = torch.nonzero(table <= ths)
            # #print(y.shape[0])
            #     count[ths]+=y.shape[0]

            #global_loss+=loss*batch
            #print()
            #global_surface_loss+=loss_1

            #global_sphere_loss+=torch.sqrt(loss_6/lamda6)
            #print(global_sphere_loss)
            #part_Loss+=loss_5

            # print('total loss : {},  surface loss : {},  surface2 loss: {}, part loss :{} ,this epoch take {}s'. \
            #       format(global_loss/(step+1),global_surface_loss/(step+1),global_sphere_loss/(step+1),part_Loss/(step+1),time.time()-time0))
            print(
                'PV : {},  PJ : {}, input_error: {} ,P_input_error: {} this epoch take {}s'. \
                format(PV / (step+1), PJ/(step+1),J_error/(step+1),P_J_error/(step+1), time.time() - time0))
    # loss_count=loss_count[1:-1,:].squeeze()
    # loss_sort, idx1 = torch.sort(loss_count, descending=True)
    # idx = idx1[0:20]
    # print(idx)
    # torch.save(loss_sort, 'loss_sort.pth')
    # torch.save(idx1,'loss_count_idx.pth')

    #count/=(35855*2000)
    #print(count)