
import os
import threading, queue
import numpy as np
import time


if __name__ == '__main__':

	################  per scene NeRF  ################
	commands = {
			'-grid': '', \
			# '-DVGO-like': 'model.basis_type=none model.coeff_reso=80', \
			# '-noC': 'model.coeff_type=none', \
			# '-SL':'model.basis_dims=[18]  model.basis_resos=[70] model.freq_bands=[8.]', \
			# '-CP': f'model.coeff_type=vec model.basis_type=cp model.freq_bands=[1.,1.,1.,1.,1.,1.] model.basis_resos=[512,512,512,512,512,512] model.basis_dims=[32,32,32,32,32,32]', \
			# '-iNGP-like': 'model.basis_type=hash  model.coeff_type=none', \
			# '-hash': f'model.basis_type=hash model.coef_init=1.0', \
			# '-sinc': f'model.basis_mapping=sinc', \
			# '-tria': f'model.basis_mapping=triangle', \
			# '-vm': f'model.coeff_type=vm model.basis_type=vm', \
			# '-mlpB': 'model.basis_type=mlp', \
			# '-mlpC': 'model.coeff_type=mlp', \
			# '-occNet': f'model.basis_type=x model.coeff_type=none model.basis_mapping=x model.num_layers=8 model.hidden_dim=256 ', \
			# '-nerf': f'model.basis_type=x model.coeff_type=none model.basis_mapping=trigonometric ' \
			# 		f'model.num_layers=8 model.hidden_dim=256 ' \
			# 		f'model.freq_bands=[1.,2.,4.,8.,16.,32.,64,128,256.,512.] model.basis_dims=[1,1,1,1,1,1,1,1,1,1] model.basis_resos=[1024,512,256,128,64,32,16,8,4,2]', \
			# '-iNGP-like-sl': 'model.basis_type=hash  model.coeff_type=none model.basis_dims=[16] model.freq_bands=[8.] model.basis_resos=[64] ', \
			# '-hash-sl': f'model.basis_type=hash model.coef_init=1.0 model.basis_dims=[16] model.freq_bands=[8.] model.basis_resos=[64] ', \
		# '-DCT':'model.basis_type=fix-grid', \
		}

	################  per scene NeRF  ################
	######  uncomment the following five lines if you want to train on all scenes #########
	cmds = []
	for name in commands.keys(): #
		# for scene in ['ship', 'mic', 'chair', 'lego', 'drums', 'ficus', 'hotdog', 'materials']:#
		# 	cmd = f'python train_per_scene.py configs/nerf.yaml defaults.expname={scene}{name} dataset.datadir=./data/nerf_synthetic/{scene} {commands[name]}'
		# 	cmds.append(cmd)

		for scene in ['Ignatius','Truck']:#
			if scene != 'Ignatius':
				cmd = f'python train_per_scene.py configs/nerf.yaml defaults.expname={scene}{name} dataset.datadir=./data/TanksAndTemple/{scene} {commands[name]} ' \
					f' dataset.dataset_name=tankstemple '
				cmds.append(cmd)
		
			cmd = f'python train_per_scene.py configs/nerf.yaml defaults.expname={scene}{name} dataset.datadir=./data/TanksAndTemple/{scene} {commands[name]} ' \
				f' dataset.dataset_name=tankstemple exportation.render_only=1 exportation.render_path=1 exportation.render_test=0 ' \
				f' defaults.ckpt=/mnt/qb/home/geiger/zyu30/Projects/Anpei/Code/factor-fields/logs/{scene}-grid/{scene}-grid.th '
			cmds.append(cmd)

	################  generalization NeRF  ################
	commands = {
			# '-grid': '', \
			# '-DVGO-like': 'model.basis_type=none model.coeff_reso=48',
			# '-SL':'model.basis_dims=[72]  model.basis_resos=[48] model.freq_bands=[6.]', \
			# '-CP': f'model.coeff_type=vec model.basis_type=cp model.freq_bands=[1.,1.,1.,1.,1.,1.] model.basis_resos=[512,512,512,512,512,512] model.basis_dims=[32,32,32,32,32,32]', \
			# '-hash': f'model.basis_type=hash model.coef_init=1.0 ', \
			# '-sinc': f'model.basis_mapping=sinc', \
			# '-tria': f'model.basis_mapping=triangle', \
			# '-vm': f'model.coeff_type=vm model.basis_type=vm', \
			# '-mlpB': 'model.basis_type=mlp', \
			# '-mlpC': 'model.coeff_type=mlp', \
			# '-hash-sl': f'model.basis_type=hash model.coef_init=1.0 model.basis_dims=[16] model.freq_bands=[8.] model.basis_resos=[64] ', \
		# '-DCT':'model.basis_type=fix-grid', \
		}
	# for name in commands.keys(): #
	# 	config = commands[name]
	# 	config = f'python train_across_scene2.py configs/nerf_set.yaml defaults.expname=google-obj{name} {config} ' \
	# 			f'training.volume_resoFinal=128 dataset.datadir=./data/google_scanned_objects/'
	# 	cmds.append(config)


	# # =========> fine tuning <================
	# views = 5
	# for name in commands.keys():  #
	# 	for scene in [183,199,298,467,957,244,963,527]:#
	# 		cmd = f'python train_across_scene.py configs/nerf_ft.yaml defaults.expname=google_objs_{name}_{scene}_{views}_views ' \
	# 			f'dataset.datadir=/home/anpei/Dataset/google_scanned_objects/ {commands[name]} ' \
	# 			f'dataset.train_views={views} ' \
	# 			f'dataset.train_scene_list=[{scene}] ' \
	# 			f'dataset.test_scene_list=[{scene}] ' \
	# 			f'defaults.ckpt=/home/anpei/Code/NeuBasis/log/google-obj{name}//google-obj{name}.th  '
	# 		cmds.append(cmd)

	# for scene in ['Ignatius','Barn','Truck','Family','Caterpillar']:#'Ignatius','Barn','Truck','Family','Caterpillar'
	# 	cmds.append(f'python train_basis.py configs/tnt.yaml defaults.expname=tnt_{scene} ' \
	# 		f'dataset.datadir=./data/TanksAndTemple/{scene}'
	# 		)

	# cmds = []
	# for scene in ['room']:#,'hall','kitchen','living_room','room2','sofa','meeting_room','room','salon2'
	# 	cmds.append(f'python train_basis.py configs/colmap_new.yaml defaults.expname=indoor_{scene} ' \
	# 		f'dataset.datadir=./data/indoor/real/{scene}'
	# 		# f'defaults.ckpt=/home/anpei/code/NeuBasis2/log/basis_ship/basis_ship.th exporation.render_only=True'
	# 		)

	# cmds = []
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeRF model.basis_type=x model.coeff_type=none model.basis_mapping=trigonometric ' \
	# 			f'model.num_layers=8 model.hidden_dim=256 ' \
	# 			f'model.freq_bands=[1.,2.,4.,8.,16.,32.,64,128,256.,512.] model.basis_dims=[1,1,1,1,1,1,1,1,1,1] model.basis_resos=[1024,512,256,128,64,32,16,8,4,2]')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-grid')
	# cmds.append(f'python 2D_regression.py  defaults.expname=NeuBasis-mlpB model.basis_type=mlp')
	# cmds.append(f'python 2D_regression.py  defaults.expname=NeuBasis-mlpC model.coeff_type=mlp')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=DVGO-like model.basis_type=none')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-noC model.coeff_type=none')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-sinc model.basis_mapping=sinc')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-tria model.basis_mapping=triangle')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-SL model.basis_dims=[144]  model.basis_resos=[14] model.freq_bands=[73.14]')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-DCT model.basis_type=fix-grid')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-CP model.coeff_type=vec model.basis_type=cp \
	# 				model.freq_bands=[1.,1.,1.,1.,1.,1.] model.basis_resos=[1024,1024,1024,1024,1024,1024] model.basis_dims=[64,64,64,32,32,32]')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=iNGP-like model.basis_type=hash  model.coeff_type=none')
	# cmds.append(f'python 2D_regression.py configs/image.yaml defaults.expname=NeuBasis-hash model.basis_type=hash model.coef_init=0.1 basis_dims=[16,16,16,16,16,16]')

	#setting available gpus
	gpu_idx = [0]
	gpus_que = queue.Queue(len(gpu_idx))
	for i in gpu_idx:
		gpus_que.put(i)

	# os.makedirs(f"log/{expFolder}", exist_ok=True)
	def run_program(gpu, cmd):
		cmd = f'{cmd} '
		print(cmd)
		os.system(cmd)
		gpus_que.put(gpu)


	ths = []
	for i in range(len(cmds)):

		gpu = gpus_que.get()
		t = threading.Thread(target=run_program, args=(gpu, cmds[i]), daemon=True)
		t.start()
		ths.append(t)

	for th in ths:
		th.join()


# import os 
# import numpy as np
# root = f'/mnt/qb/home/geiger/zyu30/Projects/Anpei/Code/factor-fields/logs/'
# # root = '/cluster/home/anchen/root/Code/NeuBasis/log/'
# scores = []
# # for scene in ['ship', 'mic', 'chair', 'lego', 'drums', 'ficus', 'hotdog', 'materials']:
# for scene in ['Caterpillar','Family','Ignatius','Truck']:
# 	scores.append(np.loadtxt(f'{root}/{scene}-grid/imgs_test_all/mean.txt'))
# 	# os.system(f'cp {root}/{scene}-grid/imgs_test_all/video.mp4 /mnt/qb/home/geiger/zyu30/Projects/Anpei/Code/factor-fields/logs/video/{scene}.mp4')
# 	os.system(f'cp {root}/{scene}-grid/{scene}-grid/imgs_path_all/video.mp4 /mnt/qb/home/geiger/zyu30/Projects/Anpei/Code/factor-fields/logs/video/{scene}.mp4')
# # print(np.mean(np.stack(scores),axis=0))