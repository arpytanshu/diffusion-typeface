## Denoising Diffusion Implementation from Scratch
 Minimal implementation of DDPM w/ CFG from scratch.  
 Trained on English typefaces dataset from `https://www.kaggle.com/datasets/killen/bw-font-typefaces`  
 Capable of conditional generation of varied English Typefaces.  

![Diffusion](assets/diffusion3.gif)

<br>  




### Training
Put each class image in a seperate directory, and update path of parent directory in training script:
```
args.dataset_path = <path to dataset>
args.num_classes = <number of classes in dataset>
args.run_name = <some cool name>
```
And everything should hopefully work fine.

<br>  




### Generation
```
python generations.py --run_name DDPM_cond1 --plot_save_path /tmp --string demonstration -cfg_scale 5
```

<br>  




### Sample Generation
![Demo](assets/demo.gif)

