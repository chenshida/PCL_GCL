this is a DLP API for python 
- requirement
```
hidapi==0.7.99.post21
numpy==1.16.0
hid==1.0.0
```
- install 
```
python setup.py install 
```

- sample
```
import DLP4500
test_api = DLP4500.DLP4500API()
test_api.loadImageFromFlash(10)
```
it will load image index 10 from flash and reproject, the api need use USB, maybe it need admin permission.