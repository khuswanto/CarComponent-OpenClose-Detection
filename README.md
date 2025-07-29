# CarComponent-OpenClose-Detection

## Installation
1. Install all the python required libraries
   ```shell
   pip install -r requirements.txt
   ```
2. Install also flash-attention2
   ```shell
   pip install flash-attn
   ```
   But, for window user like me, download the wheel file from: [link](https://github.com/abshkd/flash-attention-windows/releases)
   ```shell
   pip install "flash_attn-2.7.4-cp312-cp312-win_amd64.whl"
   ```

## Run
Run command below with desired objective number. `1` for deep learning and `2` for vision languange model.
```shell
python main.py 1
```
