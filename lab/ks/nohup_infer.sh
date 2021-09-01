python inference.py --name age_cls --num_classes 3 --model KSResnetModel --batch_size 16 --augmentation CustomAugmentation 
python inference.py --name gen_cls --num_classes 2 --model KSResnetModel --batch_size 16 --augmentation CustomAugmentation 
python inference.py --name mask_cls --num_classes 3 --model KSResnetModel --batch_size 16 --augmentation CustomAugmentation 