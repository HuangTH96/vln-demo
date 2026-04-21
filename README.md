# 测试
```
cd vln-demo/tests/
python -s -m pytest /test_tello_api/test_tello_api.py::<test_function_name>
```

# QuickStart
```
# airsim中仿真
cd <path/to/PX4-Autopilote>
make px4_sitl none_iris

~/UnrealEngine/Engine/Binaries/Linux/UE4Editor ~/CityParkEnvironment/CityParkEnvironment.uproject
# 打开后，在UE中点击Play

cd vln-demo/scripts
conda activate airsim
python airsim_qwen_api.py
```