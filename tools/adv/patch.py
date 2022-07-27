import torch
import torch.nn as nn
import numpy as np
import math
from math import pi
import torch.nn.functional as F
from .median_pool import MedianPool2d

class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def rect_occluding(self, num_rect=1, n_batch=8, n_feature=14, patch_size=300, with_cuda=True):
        if (with_cuda):
            device = 'cuda'
        else:
            device = 'cpu'
        tensor_img = torch.full((3, patch_size, patch_size), 0.0).to(device)
        for ttt in range(num_rect):
            xs = torch.randint(0, int(patch_size / 2), (1,))[0]
            xe = torch.randint(xs,
                               torch.min(torch.tensor(tensor_img.size()[-1]), xs + int(patch_size / 2)),
                               (1,))[0]
            ys = torch.randint(0, int(patch_size / 2), (1,))[0]
            ye = torch.randint(ys,
                               torch.min(torch.tensor(tensor_img.size()[-1]), ys + int(patch_size / 2)),
                               (1,))[0]
            tensor_img[:, xs:xe, ys:ye] = 0.5
        tensor_img_batch = tensor_img.unsqueeze(0)  ##  torch.Size([1, 3, 300, 300])
        tensor_img_batch = tensor_img_batch.expand(n_batch, n_feature, -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])
        return tensor_img_batch.to(device)

    def forward(self, adv_patch, lab_batch, img_size, patch_mask=[], by_rectangle=False, do_rotate=True, rand_loc=True,
                with_black_trans=False, scale_rate=0.2, with_crease=False, with_projection=False,
                with_rectOccluding=False, enable_empty_patch=False, enable_no_random=False, enable_blurred=True):
        # torch.set_printoptions(edgeitems=sys.maxsize)
        # print("adv_patch size: "+str(adv_patch.size()))
        # patch_size = adv_patch.size(2)

        # init adv_patch. torch.Size([3, 128, 128])
        adv_patch_size = adv_patch.size()[-1]
        if (adv_patch_size > img_size):  # > img_size(416)
            adv_patch = adv_patch.unsqueeze(0)
            adv_patch = F.interpolate(adv_patch, size=img_size)
            adv_patch = adv_patch[0]

        # st()
        # np.save('gg', adv_batch.cpu().detach().numpy())
        # gg=np.load('gg.npy')   np.argwhere(gg!=adv_batch.cpu().detach().numpy())
        def deg_to_rad(deg):
            return torch.tensor(deg * pi / 180.0).float().cuda()

        def rad_to_deg(rad):
            return torch.tensor(rad * 180.0 / pi).float().cuda()

        def get_warpR(anglex, angley, anglez, fov, w, h):
            fov = torch.tensor(fov).float().cuda()
            w = torch.tensor(w).float().cuda()
            h = torch.tensor(h).float().cuda()
            z = torch.sqrt(w ** 2 + h ** 2) / 2 / torch.tan(deg_to_rad(fov / 2)).float().cuda()
            rx = torch.tensor([[1, 0, 0, 0],
                               [0, torch.cos(deg_to_rad(anglex)), -torch.sin(deg_to_rad(anglex)), 0],
                               [0, -torch.sin(deg_to_rad(anglex)), torch.cos(deg_to_rad(anglex)), 0, ],
                               [0, 0, 0, 1]]).float().cuda()
            ry = torch.tensor([[torch.cos(deg_to_rad(angley)), 0, torch.sin(deg_to_rad(angley)), 0],
                               [0, 1, 0, 0],
                               [-torch.sin(deg_to_rad(angley)), 0, torch.cos(deg_to_rad(angley)), 0, ],
                               [0, 0, 0, 1]]).float().cuda()
            rz = torch.tensor([[torch.cos(deg_to_rad(anglez)), torch.sin(deg_to_rad(anglez)), 0, 0],
                               [-torch.sin(deg_to_rad(anglez)), torch.cos(deg_to_rad(anglez)), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().cuda()
            r = torch.matmul(torch.matmul(rx, ry), rz)
            pcenter = torch.tensor([h / 2, w / 2, 0, 0]).float().cuda()
            p1 = torch.tensor([0, 0, 0, 0]).float().cuda() - pcenter
            p2 = torch.tensor([w, 0, 0, 0]).float().cuda() - pcenter
            p3 = torch.tensor([0, h, 0, 0]).float().cuda() - pcenter
            p4 = torch.tensor([w, h, 0, 0]).float().cuda() - pcenter
            dst1 = torch.matmul(r, p1)
            dst2 = torch.matmul(r, p2)
            dst3 = torch.matmul(r, p3)
            dst4 = torch.matmul(r, p4)
            list_dst = [dst1, dst2, dst3, dst4]
            org = torch.tensor([[0, 0],
                                [w, 0],
                                [0, h],
                                [w, h]]).float().cuda()
            dst = torch.zeros((4, 2)).float().cuda()
            for i in range(4):
                dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
                dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
            org = org.unsqueeze(0)
            dst = dst.unsqueeze(0)
            warpR = tgm.get_perspective_transform(org, dst).float().cuda()
            return warpR

        ## get y gray
        # adv_patch_yuv = Colorspace("rgb", "yuv")(adv_patch).cuda()
        # y = adv_patch_yuv[0].unsqueeze(0)
        # adv_patch_new_y_gray = torch.cat((y,y,y), 0).cuda()
        ## get   gray
        # y = (0.2989 * adv_patch[0] + 0.5870 * adv_patch[1] + 0.1140 * adv_patch[2]).unsqueeze(0)
        # adv_patch_new_y_gray = torch.cat((y,y,y), 0).cuda()
        # adv_patch = adv_patch_new_y_gray

        def warping(input_tensor_img, wrinkle_p=15):
            C, H, W = input_tensor_img.size()
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # torch.Size([2, H, W])
            # print("grid "+str(grid.shape)+" : \n"+str(grid))
            grid = grid.view(2, -1)  # torch.Size([2, H*W])
            grid = grid.permute(1, 0)  # torch.Size([H*W, 2])
            perturbed_mesh = grid

            # nv = np.random.randint(20) - 1
            nv = wrinkle_p
            for k in range(nv):
                # Choosing one vertex randomly
                vidx = np.random.randint(grid.shape[0])
                vtex = grid[vidx, :]
                # Vector between all vertices and the selected one
                xv = perturbed_mesh - vtex
                # Random movement
                mv = (np.random.rand(1, 2) - 0.5) * 20
                hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] + 1))
                hxv[:, :-1] = xv
                hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0], 1))
                d = np.cross(hxv, hmv)
                d = np.absolute(d[:, 2])
                # print("d "+str(d.shape)+" :\n"+str(d))
                d = d / (np.linalg.norm(mv, ord=2))
                wt = d
                curve_type = np.random.rand(1)
                if curve_type > 0.3:
                    alpha = np.random.rand(1) * 50 + 50
                    wt = alpha / (wt + alpha)
                else:
                    alpha = np.random.rand(1) + 1
                    wt = 1 - (wt / 100) ** alpha
                msmv = mv * np.expand_dims(wt, axis=1)
                perturbed_mesh = perturbed_mesh + msmv

            perturbed_mesh_2 = perturbed_mesh.permute(1, 0)
            max_x = torch.max(perturbed_mesh_2[0])
            min_x = torch.min(perturbed_mesh_2[0])
            # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
            max_y = torch.max(perturbed_mesh_2[1])
            min_y = torch.min(perturbed_mesh_2[1])
            # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
            perturbed_mesh_2[0, :] = (W - 1) * (perturbed_mesh_2[0, :] - min_x) / (max_x - min_x)
            perturbed_mesh_2[1, :] = (H - 1) * (perturbed_mesh_2[1, :] - min_y) / (max_y - min_y)
            # max_x = torch.max(perturbed_mesh_2[0])
            # min_x = torch.min(perturbed_mesh_2[0])
            # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
            # max_y = torch.max(perturbed_mesh_2[1])
            # min_y = torch.min(perturbed_mesh_2[1])
            # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
            perturbed_mesh_2 = perturbed_mesh_2.contiguous().view(-1, H, W).float()
            # print("perturbed_mesh_2 dtype : "+str(perturbed_mesh_2.data.type()))
            # print("perturbed_mesh_2 "+str(perturbed_mesh_2.shape)+" : \n"+str(perturbed_mesh_2))

            vgrid = perturbed_mesh_2.unsqueeze(0).cuda()
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0  # max(W-1,1)
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0  # max(H-1,1)
            vgrid = vgrid.permute(0, 2, 3, 1).cuda()
            input_tensor_img_b = input_tensor_img.unsqueeze(0).cuda()
            output = F.grid_sample(input_tensor_img_b, vgrid, align_corners=True)  # torch.Size([1, 3, H, W])
            return output[0]

        if (with_crease):
            # warping
            adv_patch = warping(adv_patch)  # torch.Size([3, H, W])
            # print("adv_patch "+str(adv_patch.size())+"  "+str(adv_patch.dtype))

        #
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        if (enable_blurred):
            adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        else:
            adv_patch = adv_patch.unsqueeze(0)
        # print("adv_patch medianpooler size: "+str(adv_patch.size())) ## torch.Size([1, 3, 300, 300])
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  ##  torch.Size([1, 1, 3, 300, 300])
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                     -1)  ##  torch.Size([8, 14, 3, 300, 300])
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        if not (len(patch_mask) == 0):
            ## mask size : torch.Size([3, 300, 300])
            patch_mask = patch_mask.unsqueeze(0)  ## mask size : torch.Size([1, 3, 300, 300])
            mask_batch = patch_mask.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                           -1)  ## mask size : torch.Size([8, 14, 3, 300, 300])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size()))  ##  contrast size : torch.Size([8, 14, 3, 300, 300])

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([8, 14, 3, 300, 300])

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([8, 14, 3, 300, 300])
        # print(noise[0,0,0,:10,0])
        # # Apply contrast/brightness/noise, clamp

        # print("adv_patch  : "+str(adv_patch.is_cuda))
        # print("adv_batch  : "+str(adv_batch.is_cuda))
        # print("contrast   : "+str(contrast.is_cuda))
        # print("brightness : "+str(brightness.is_cuda))
        # print("noise      : "+str(noise.is_cuda))

        ## adv_patch 已經模糊
        if enable_no_random and not (enable_empty_patch):
            adv_batch = adv_batch
        if not (enable_no_random) and not (enable_empty_patch):
            adv_batch = adv_batch * contrast + brightness + noise
        if not (len(patch_mask) == 0):
            adv_batch = adv_batch * mask_batch
        if (with_rectOccluding):
            rect_occluder = self.rect_occluding(num_rect=2, n_batch=adv_batch.size()[0], n_feature=adv_batch.size()[1],
                                                patch_size=adv_batch.size()[-1])
            adv_batch = torch.where((rect_occluder == 0), adv_batch, rect_occluder)

        # # get   gray
        # # print("adv_batch size: "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # # adv_batch = adv_batch.cpu()
        # adv_batch_r = adv_batch[:,:,0,:,:]  ##  torch.Size([3, 300, 300])
        # adv_batch_g = adv_batch[:,:,1,:,:]  ##  torch.Size([3, 300, 300])
        # adv_batch_b = adv_batch[:,:,2,:,:]  ##  torch.Size([3, 300, 300])
        # # print("adv_batch_r size: "+str(adv_batch_r.size()))  ##  torch.Size([8, 14, 300, 300])
        # y = (0.2989 * adv_batch_r + 0.5870 * adv_batch_g + 0.1140 * adv_batch_b)
        # y = y.unsqueeze(2)
        # # print("y size: "+str(y.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # adv_batch_new_y_gray = torch.cat((y,y,y), 2).cuda()
        # adv_batch = adv_batch_new_y_gray
        # # print("adv_batch size: "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # # adv_batch = adv_batch.cuda()

        #
        if (with_black_trans):
            adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
        else:
            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        adv_patch_set = adv_batch[0, 0]

        # ## split img
        # # print("adv_batch size : "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # adv_batch_units = []
        # split_side = 2
        # split_step = patch_size / split_side
        # # print("split_step : "+str(split_step))
        # for stx in range(0, split_side):
        #     for sty in range(0, split_side):
        #         x_s = int(0 + stx*split_step)
        #         y_s = int(0 + sty*split_step)
        #         x_e = int(x_s+split_step)
        #         y_e = int(y_s+split_step)
        #         adv_batch_unit = adv_batch[:,:,:, x_s:x_e, y_s:y_e].cuda()
        #         adv_batch_zeroes = torch.zeros(adv_batch.size()).cuda()
        #         adv_batch_zeroes[:,:,:, x_s:x_e, y_s:y_e] = adv_batch_unit
        #         adv_batch_unit = adv_batch_zeroes
        #         adv_batch_units.append(adv_batch_unit)

        def resize_rotate(adv_batch, by_rectangle=False):

            if (with_black_trans):
                adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
            else:
                adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

            # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
            cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8, 14, 1])
            cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
            cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
            cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
            cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
            cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - msk_batch.size(-1)) / 2  # (416-300) / 2 = 58
            # print("pad : "+str(pad))
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            # print("adv_batch size : "+str(adv_batch.size()))
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            msk_batch = mypad(msk_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            # print("adv_batch size : "+str(adv_batch.size()))

            # Rotation and rescaling transforms
            anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 8*14 = 112
            if do_rotate:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)  # torch.Size([112])
            else:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)

            # Resizes and rotates
            current_patch_size = adv_patch.size(-1)
            print(lab_batch[0])
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
            lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
            lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
            lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
            lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
            target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(scale_rate)) ** 2) + (
                        (lab_batch_scaled[:, :, 4].mul(scale_rate)) ** 2))  # torch.Size([8, 14])
            target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            print(lab_batch_scaled[0], target_x[0], target_y[0])
            if (rand_loc):
                off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                target_x = target_x + off_x
                off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
                target_y = target_y + off_y
            target_y = target_y - 0.05
            print("current_patch_size : " + str(current_patch_size))
            print("target_size        : " + str(target_size.size()))
            print("target_size        : " + str(target_size))
            scale = target_size / current_patch_size  # torch.Size([8, 14])
            scale = scale.view(anglesize)  # torch.Size([112]) 8*14
            # print("scale : "+str(scale))

            s = adv_batch.size()
            adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
            msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])

            tx = (-target_x + 0.5) * 2
            ty = (-target_y + 0.5) * 2
            sin = torch.sin(angle)
            cos = torch.cos(angle)

            # Theta = rotation,rescale matrix
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
            theta[:, 0, 0] = (cos / scale)
            theta[:, 0, 1] = sin / scale
            theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)
            theta[:, 1, 0] = -sin / scale
            theta[:, 1, 1] = (cos / scale)
            theta[:, 1, 2] = (-tx * sin / scale + ty * cos / scale)

            if (by_rectangle):
                theta[:, 1, 1] = theta[:, 1, 1] / 1.5
                theta[:, 1, 2] = theta[:, 1, 2] / 1.5
            # print(tx)
            # print(theta[:, 0, 2])
            # print(1*cos/scale)
            # print(-1*cos/scale)

            # print("theta :\n"+str(theta))
            # sys.exit()

            b_sh = adv_batch.shape  # b_sh = torch.Size([112, 3, 416, 416])
            grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([112, 416, 416, 2])

            adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([112, 3, 416, 416])
            msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([112, 3, 416, 416])

            # print("grid : "+str(grid[0,200:300,200:300,:]))

            # msk_batch_t_r = msk_batch_t[:,0,:,:]
            # msk_batch_t_g = msk_batch_t[:,0,:,:]
            # msk_batch_t_b = msk_batch_t[:,0,:,:]
            # for t in range(msk_batch_t.size()[0]):
            #     dx = int(grid[t,0,0,0])
            #     dx2 = int(grid[t,400,400,0])
            #     dy = int(grid[t,0,0,1])
            #     dy2 = int(grid[t,400,400,1])
            #     msk_batch_t[t,0,dx:dx2,dy:dy2] = 0
            #     msk_batch_t[t,1,dx:dx2,dy:dy2] = 0
            #     msk_batch_t[t,2,dx:dx2,dy:dy2] = 0

            # # angle 2
            # tx = (-target_x+0.5)*2
            # ty = (-target_y+0.5)*2
            # sin = torch.sin(angle)
            # cos = torch.cos(angle)

            # # Theta = rotation,rescale matrix
            # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
            # theta[:, 0, 0] = cos/scale
            # theta[:, 0, 1] = sin/scale
            # theta[:, 0, 2] = 0
            # theta[:, 1, 0] = -sin/scale
            # theta[:, 1, 1] = cos/scale
            # theta[:, 1, 2] = 0

            '''
            # Theta2 = translation matrix
            theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
            theta2[:, 0, 0] = 1
            theta2[:, 0, 1] = 0
            theta2[:, 0, 2] = (-target_x + 0.5) * 2
            theta2[:, 1, 0] = 0
            theta2[:, 1, 1] = 1
            theta2[:, 1, 2] = (-target_y + 0.5) * 2

            grid2 = F.affine_grid(theta2, adv_batch.shape)
            adv_batch_t = F.grid_sample(adv_batch_t, grid2)
            msk_batch_t = F.grid_sample(msk_batch_t, grid2)

            '''
            adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
            msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])

            if (with_black_trans):
                adv_batch_t = torch.clamp(adv_batch_t, 0.0, 0.99999)
            else:
                adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.99999)
            # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
            # img = transforms.ToPILImage()(img)
            # img.show()
            # exit()

            # output: torch.Size([8, 14, 3, 416, 416]), torch.Size([8, 14, 3, 416, 416])
            # return adv_batch_t * msk_batch_t, (adv_batch_t * msk_batch_t0), (adv_batch_t * msk_batch_t1), (adv_batch_t * msk_batch_t2),  (adv_batch_t * msk_batch_t3), adv_batch_t, msk_batch_t
            return (adv_batch_t * msk_batch_t), msk_batch_t

        # adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t = resize_rotate(adv_batch)
        adv_batch_masked, msk_batch = resize_rotate(adv_batch,
                                                    by_rectangle)  # adv_batch torch.Size([8, 7, 3, 150, 150])   adv_batch_masked torch.Size([8, 7, 3, 416, 416])

        if (with_projection):
            adv_batch = adv_batch_masked
            # # Rotating a Image
            b, f, c, h, w = adv_batch.size()
            adv_batch = adv_batch.view(b * f, c, h, w)
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            batch, channel, width, height = adv_batch.size()
            padding_borader = torch.nn.ZeroPad2d(50)
            input_ = padding_borader(adv_batch)
            # print("input_ "+str(input_.size())+"  "+str(input_.dtype))
            angle = np.random.randint(low=-50, high=51)
            mat = get_warpR(anglex=0, angley=angle, anglez=0, fov=42, w=width, h=height)
            mat = mat.expand(batch, -1, -1, -1)
            # print("image  "+str(self.image.dtype)+"  "+str(self.image.size()))
            # print("input_ "+str(input_.dtype)+"  "+str(input_.size()))
            # print("mat    "+str(mat.dtype)+"  "+str(mat.size()))
            adv_batch = tgm.warp_perspective(input_, mat, (input_.size()[-2], input_.size()[-1]))
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            adv_batch = adv_batch.view(b, f, c, input_.size()[-2], input_.size()[-1])
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            ##
            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - adv_batch.size(-1)) / 2  # (416-300) / 2 = 58
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            adv_batch_masked = adv_batch

        # adv_batch_masked = torch.clamp(adv_batch_masked, 0.0, 0.99999)
        # return adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t, adv_patch_set
        return adv_batch_masked, adv_patch_set, msk_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        # print("img_batch size : "+str(img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
        # print("adv_batch size : "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 416, 416])
        advs = torch.unbind(adv_batch, 1)
        # print("advs (np) size : "+str(np.array(advs).shape))  ##  (14,)
        # print("b[0].size      : "+str(b[0].size()))  ##  torch.Size([8, 3, 416, 416])
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch