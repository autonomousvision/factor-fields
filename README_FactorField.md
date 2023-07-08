## nerf reconstruction with Dictionary field

```python
	for scene in ['ship', 'mic', 'chair', 'lego', 'drums', 'ficus', 'hotdog', 'materials']:
		cmd = f'python train_basis.py configs/nerf.yaml defaults.expname={scene} ' \
			    f'dataset.datadir=./data/nerf_synthetic/{scene} ' 
```

## different model design choices

```python
	choice_dict = {
			'-grid': '', \
			'-DVGO-like': 'model.basis_type=none model.coeff_reso=80', \
			'-noC': 'model.coeff_type=none', \
			'-SL':'model.basis_dims=[18]  model.basis_resos=[70] model.freq_bands=[8.]', \
			'-CP': f'model.coeff_type=vec model.basis_type=cp model.freq_bands=[1.,1.,1.,1.,1.,1.] model.basis_resos=[512,512,512,512,512,512] model.basis_dims=[32,32,32,32,32,32]', \
			'-iNGP-like': 'model.basis_type=hash  model.coeff_type=none', \
			'-hash': f'model.basis_type=hash model.coef_init=1.0 ', \
			'-sinc': f'model.basis_mapping=sinc', \
			'-tria': f'model.basis_mapping=triangle', \
			'-vm': f'model.coeff_type=vm model.basis_type=vm', \
			'-mlpB': 'model.basis_type=mlp', \
			'-mlpC': 'model.coeff_type=mlp', \
			'-occNet': f'model.basis_type=x model.coeff_type=none model.basis_mapping=x model.num_layers=8 model.hidden_dim=256 ', \
			'-nerf': f'model.basis_type=x model.coeff_type=none model.basis_mapping=trigonometric ' \
					f'model.num_layers=8 model.hidden_dim=256 ' \
					f'model.freq_bands=[1.,2.,4.,8.,16.,32.,64,128,256.,512.] model.basis_dims=[1,1,1,1,1,1,1,1,1,1] model.basis_resos=[1024,512,256,128,64,32,16,8,4,2]', \
			'-hash-sl': f'model.basis_type=hash model.coef_init=1.0 model.basis_dims=[16] model.freq_bands=[8.] model.basis_resos=[64] ', \
			'-vm-sl': f'model.coeff_type=vm model.basis_type=vm model.coef_init=1.0 model.basis_dims=[18] model.freq_bands=[1.] model.basis_resos=[64] model.total_params=1308416 ', \
		'-DCT':'model.basis_type=fix-grid', \
		}

	for name in choice_dict.keys(): 
		for scene in [ 'ship', 'mic', 'chair', 'lego', 'drums', 'ficus', 'hotdog', 'materials']:

			cmd = f"python train_per_scene.py configs/nerf.yaml defaults.expname={scene}{name} dataset.datadir=./data/nerf_synthetic/{scene} {config}"
```

## generalized nerf
Your can choice of the the following design choice for testing.
```python
	choice_dict = {
			'-grid': '', \
			'-DVGO-like': 'model.basis_type=none model.coeff_reso=48',
			'-SL':'model.basis_dims=[72]  model.basis_resos=[48] model.freq_bands=[6.]', \
			'-CP': f'model.coeff_type=vec model.basis_type=cp model.freq_bands=[1.,1.,1.,1.,1.,1.] model.basis_resos=[512,512,512,512,512,512] model.basis_dims=[32,32,32,32,32,32]', \
			'-hash': f'model.basis_type=hash model.coef_init=1.0 ', \
			'-sinc': f'model.basis_mapping=sinc', \
			'-tria': f'model.basis_mapping=triangle', \
			'-vm': f'model.coeff_type=vm model.basis_type=vm', \
			'-mlpB': 'model.basis_type=mlp', \
			'-mlpC': 'model.coeff_type=mlp', \
			'-hash-sl': f'model.basis_type=hash model.coef_init=1.0 model.basis_dims=[16] model.freq_bands=[8.] model.basis_resos=[64] ', \
			'-vm-sl': f'model.coeff_type=vm model.basis_type=vm model.coef_init=1.0 model.basis_dims=[18] model.freq_bands=[1.] model.basis_resos=[64] model.total_params=1308416 ', \
		  '-DCT':'model.basis_type=fix-grid', \
		}
    
	for name in choice_dict.keys(): #
		cmd = f'python train_across_scene.py configs/nerf_set.yaml defaults.expname=google-obj{name} {config} ' \
				f'training.volume_resoFinal=128 dataset.datadir=./data/google_scanned_objects/'
```

You can also fine tune of the trained model for a new scene:

```python
	for views in  [5]:#3,
		for name in choice_dict.keys():  #
			for scene in [183]:#183,199,298,467,957,244,963,527,

				cmd = f'python train_across_scene_ft.py configs/nerf_ft.yaml defaults.expname=google_objs_{name}_{scene}_{views}_views ' \
					f'{config} training.n_iters=10000 ' \
					f'dataset.train_views={views} ' \
					f'dataset.train_scene_list=[{scene}] ' \
					f'dataset.test_scene_list=[{scene}] ' \
					f'dataset.datadir=./data/google_scanned_objects/ ' \
					f'defaults.ckpt=./logs/google-obj{name}//google-obj{name}.th'
```

# render path after optimization
```python
	for views in  [5]:
		for name in choice_dict.keys():  #
			config = commands[name].replace(",", "','")
			for scene in [183]:#183,199,298,467,957,244,963,527,681,948

				cmd = f'python train_across_scene.py configs/nerf_ft.yaml defaults.expname=google_objs_{name}_{scene}_{views}_views ' \
					f'{config} training.n_iters=10000 ' \
					f'dataset.train_views={views} exporation.render_only=True exporation.render_path=True exporation.render_test=False ' \
					f'dataset.train_scene_list=[{scene}] ' \
					f'dataset.test_scene_list=[{scene}] ' \
					f'dataset.datadir=./data/google_scanned_objects/ ' \
					f'defaults.ckpt=./logs/google_objs_{name}_{scene}_{views}_views//google_objs_{name}_{scene}_{views}_views.th'
```