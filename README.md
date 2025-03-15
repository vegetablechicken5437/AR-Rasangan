# AR Rasangan Project  
這個專案使用 OpenCV 與 Mediapipe 進行即時的手部與臉部偵測，並在偵測到的部位上覆蓋 Naruto 與 Rasengan 特效。其中 Rasengan 特效會依據時間持續旋轉，達成動態視覺效果。  

![](https://github.com/vegetablechicken5437/AR-Rasangan/blob/main/rasangan_demo.gif)

### 功能特色
即時手部偵測：利用 Mediapipe 取得手部 landmark 資訊，並依據手部特徵覆蓋旋轉的 Rasengan 特效。  
即時臉部偵測：透過 Mediapipe 進行臉部偵測，並在臉部區域覆蓋 Naruto 特效。  
動態特效：Rasengan 特效會依時間持續旋轉，增加視覺動態感。  
影像處理：利用 OpenCV 處理影像合成、遮罩生成與邊界檢查。  

### 安裝與執行
```
git clone https://github.com/vegetablechicken5437/AR-Rasangan.git
cd AR-Rasangan
```

### 建立虛擬環境（選用）：
```
python -m venv venv
```
Linux/MacOS
```
source venv/bin/activate  
```
Windows
```
venv\Scripts\activate     
```

### 安裝必要套件：
```
pip install opencv-python mediapipe numpy
```
確保專案資料夾中有 naruto.png 與 rasangan.png 圖檔。  

### 執行程式：
```
python main.py
```
執行後會啟動攝影機，按 q 可退出程式。  
