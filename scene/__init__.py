#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import numpy as np
from scene.dataset_readers import SceneInfo

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], 
        fov_ratio=1 # for large fov exp
        ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print(f"""dataset path: {os.path.join(args.source_path, "transforms_train.json")}""")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, fov_ratio=fov_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # (r""" few-shot """)
        # if args.n_inputs > 0:
        #     print(f"few-shot settings: {args.n_inputs} shots")
        #     cam_infos = scene_info.train_cameras
        #     train_indices = np.linspace(0, len(cam_infos) - 1, args.n_inputs).astype(np.int32).tolist()
        #     cam_infos = [cam_infos[x] for x in train_indices]
        #     scene_info = SceneInfo(point_cloud=scene_info.point_cloud,
        #                    train_cameras=cam_infos,
        #                    test_cameras=scene_info.test_cameras,
        #                    nerf_normalization=scene_info.nerf_normalization,
        #                    ply_path=scene_info.ply_path)
        # (r""" few-shot """)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            init_point_cloud = scene_info.point_cloud
            # (r""" few-pcds """)
            # if args.max_pcds > 0:
            #     from scene.gaussian_model import BasicPointCloud
            #     N = len(init_point_cloud.points)
            #     max_pcds = min(N, args.max_pcds)
            #     indices = np.linspace(0, N - 1, max_pcds).astype(np.int32)
            #     init_point_cloud = BasicPointCloud(
            #         init_point_cloud.points[indices],
            #         init_point_cloud.colors[indices],
            #         init_point_cloud.normals[indices]
            #     )
            # (r""" few-pcds """)

            self.gaussians.create_from_pcd(init_point_cloud, self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
