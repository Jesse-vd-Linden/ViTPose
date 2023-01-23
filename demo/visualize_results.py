import cv2
import numpy as np


def keypoint_coco_wholebody_connection_and_color():
    dataset = "TopDownCocoDataset"
    # dataset = "TopDownCocoWholeBodyDataset"
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
    
    if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
        # show the results
        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                    [3, 5], [4, 6]]

        pose_link_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
        ]]
    elif dataset == "TopDownCocoWholeBodyDataset":
        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                                [8, 10], [1, 2], [0, 1], [0, 2],
                                [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                                [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                                [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                                [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                                [102, 103], [91, 104], [104, 105], [105, 106],
                                [106, 107], [91, 108], [108, 109], [109, 110],
                                [110, 111], [112, 113], [113, 114], [114, 115],
                                [115, 116], [112, 117], [117, 118], [118, 119],
                                [119, 120], [112, 121], [121, 122], [122, 123],
                                [123, 124], [112, 125], [125, 126], [126, 127],
                                [127, 128], [112, 129], [129, 130], [130, 131],
                                [131, 132]]

        pose_link_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ] + [16, 16, 16, 16, 16, 16] + [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ] + [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]
        pose_kpt_color = palette[[
                    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
                ]]

    return skeleton, pose_link_color, pose_kpt_color


def visualize_pose(img, pose_results, kpt_score_thr=0.3, radius=4, thickness=1):

    skeleton, pose_link_color, pose_kpt_color = keypoint_coco_wholebody_connection_and_color()
    for pose  in pose_results:
        # print(pose)

        keypoints = pose["keypoints"]

        for i in range(keypoints.shape[0]):
            if keypoints[i][2] < kpt_score_thr:
                continue

            x = int(keypoints[i][0])
            y = int(keypoints[i][1])

            color = pose_kpt_color[i % len(pose_kpt_color)]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
            # print(f"keypoint {i}, with color {color}")

            cv2.circle(img, (x,y), radius=radius, color=color, thickness=-1)

        for j in range(len(skeleton)):
            if keypoints[skeleton[j][0]][2] < kpt_score_thr:
                continue
            elif keypoints[skeleton[j][1]][2] < kpt_score_thr:
                continue

            kpt1 = keypoints[skeleton[j][0]]
            x1 = int(kpt1[0])
            y1 = int(kpt1[1])
            
            kpt2 = keypoints[skeleton[j][1]]
            x2 = int(kpt2[0])
            y2 = int(kpt2[1])


            # print(kpt1, kpt2)

            color = pose_link_color[j % len(pose_link_color)]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))

            cv2.line(img, [x1, y1], [x2, y2], color=color, thickness=thickness)

    return img

