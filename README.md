# unet-master
Unet network with mean teacher for altrasound image segmentation
## data preparation
structure of project
```
  --project
  	main.py
	unet.py
	dataset.py
  	altrasound.zip

all dataset you can access by email：1901684@stu.neu.edu.cn
```
## training
```
main.py:
		if __name__ == '__main__':
    		batch_size = 4
    		train(batch_size)

```
## testing
```
main.py:
	    if __name__ == '__main__':
		    student_ckpt = "student_weight path"
		    teacher_ckpt = "teacher_weight path"
		    test(student_ckpt, teacher_ckpt)

