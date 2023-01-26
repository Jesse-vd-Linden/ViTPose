demo:
	python3 demo/webcam_demo.py
hrnet_48:
	python3 demo/webcam_demo.py --human-pose-checkpoint ./pytorch-checkpoint-models/pose_hrnet_w48_256x192.pth --human-pose-config ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_dark.py
hrnet_32:
	python3 demo/webcam_demo.py --human-pose-checkpoint ./pytorch-checkpoint-models/pose_hrnet_w32_256x192.pth --human-pose-config ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_dark.py
vitpose_small_coco:
	python3 demo/webcam_demo.py --human-pose-checkpoint ./pytorch-checkpoint-models/vitpose_small.pth --human-pose-config ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py
vitpose_base_coco:
	python3 demo/webcam_demo.py --human-pose-checkpoint ./pytorch-checkpoint-models/vitpose-b.pth --human-pose-config ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py
vitpose_large_coco:
	python3 demo/webcam_demo.py --human-pose-checkpoint ./pytorch-checkpoint-models/vitpose-l.pth --human-pose-config ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py
vitpose_huge_coco: # too much memory for my GPU of 4GB
	python3 demo/webcam_demo.py --human-pose-checkpoint ./pytorch-checkpoint-models/vitpose-h.pth --human-pose-config ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py
vipnas_coco_whole:
	python3 demo/webcam_demo.py