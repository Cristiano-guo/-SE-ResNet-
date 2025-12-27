import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import os


# 1. 参数设置


MORANDI_COLORS = {
    'blue': '#7c8b9d',       
    'red': '#c68989',       
    'green': '#9bb0a5',      
    'bg': '#fcfcfc',       
    'grid': '#e8e8e8',       
    'text': '#555555'        
}

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'text.color': MORANDI_COLORS['text'],
    'axes.labelcolor': MORANDI_COLORS['text'],
    'xtick.color': MORANDI_COLORS['text'],
    'ytick.color': MORANDI_COLORS['text'],
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': MORANDI_COLORS['grid'],
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'savefig.dpi': 300
})

CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 60,            # 训练轮数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 0,        # 自动填充
    'signal_len': 0,         # 自动填充
    'channels': 2,
    'tta_steps': 5           # TTA次数
}

print(f"Using device: {CONFIG['device']}")

# 2. 数据集类 
class SEIDataset(Dataset):
    def __init__(self, features, labels=None, mode='train', augment=False):
        self.mode = mode
        self.augment = augment
        self.signal_len = CONFIG['signal_len']
        self.channels = CONFIG['channels']
        
        # 维度重塑
        try:
            self.features = torch.FloatTensor(features).view(-1, self.channels, self.signal_len)
        except RuntimeError:
            raise ValueError(f"数据维度不匹配，无法重塑。")
            
        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        
        # 训练增强：噪声 + 缩放
        if self.mode == 'train' and self.augment:
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
            
        # TTA 增强：仅微弱噪声
        if self.mode == 'test_tta':
             noise = torch.randn_like(x) * 0.002
             x = x + noise
            
        if self.labels is not None:
            return x, self.labels[idx]
        else:
            return x

# 3. 数据加载与清洗 (Robust Loader)
def load_data():
    print("正在加载数据...")
    
    # 路径检查
    train_path = 'ml_homework/train.csv' if os.path.exists('ml_homework/train.csv') else 'train.csv'
    test_path = 'ml_homework/test.csv' if os.path.exists('ml_homework/test.csv') else 'test.csv'

    # 读取训练集
    train_df = pd.read_csv(train_path, header=None, low_memory=False)
    
    # 清洗：强制转数字，去除表头行
    last_col_idx = train_df.shape[1] - 1
    train_df.iloc[:, last_col_idx] = pd.to_numeric(train_df.iloc[:, last_col_idx], errors='coerce')
    train_df.dropna(subset=[last_col_idx], inplace=True)
    
    y = train_df.iloc[:, last_col_idx].astype(int).values
    X = train_df.iloc[:, :-1].values.astype(float)
    
    # 自动更新配置
    CONFIG['signal_len'] = X.shape[1] // 2
    CONFIG['num_classes'] = len(np.unique(y))
    print(f"检测到数据: 信号长度={CONFIG['signal_len']}, 类别数={CONFIG['num_classes']}")

    # 归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    train_dataset = SEIDataset(X_train, y_train, mode='train', augment=True)
    val_dataset = SEIDataset(X_val, y_val, mode='val', augment=False)
    
    # 读取测试集
    test_df = pd.read_csv(test_path, header=None, low_memory=False)
    try:
        float(test_df.iloc[0, 0])
    except ValueError:
        test_df = pd.read_csv(test_path, header=0, low_memory=False)
        
    # 提取测试特征
    if test_df.shape[1] > X.shape[1]:
        X_test_raw = test_df.iloc[:, :-1].values
    else:
        X_test_raw = test_df.values
    
    # 清洗测试集并归一化
    X_test_raw = pd.DataFrame(X_test_raw).apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_test = scaler.transform(X_test_raw)
    
    return train_dataset, val_dataset, X_test, test_df


# 4. SE-ResNet1D 
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SEResNet1D(nn.Module):
    def __init__(self):
        super(SEResNet1D, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, CONFIG['num_classes'])
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x

# 5. 可视化绘图
def plot_history_morandi(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Loss 曲线
    ax1.plot(epochs, history['train_loss'], label='Train Loss', 
             color=MORANDI_COLORS['blue'], linewidth=2.5, alpha=0.9)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', 
             color=MORANDI_COLORS['red'], linewidth=2.5, linestyle='--', alpha=0.9)
    ax1.fill_between(epochs, history['train_loss'], history['val_loss'], 
                     color=MORANDI_COLORS['blue'], alpha=0.1)
    
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)
    sns.despine(ax=ax1) 

    # 2. Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], label='Train Acc', 
             color=MORANDI_COLORS['green'], linewidth=2.5, alpha=0.9)
    ax2.plot(epochs, history['val_acc'], label='Val Acc', 
             color=MORANDI_COLORS['blue'], linewidth=2.5, linestyle='--', alpha=0.9)
    ax2.fill_between(epochs, history['train_acc'], history['val_acc'], 
                     color=MORANDI_COLORS['green'], alpha=0.1)

    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(frameon=True, fancybox=True, framealpha=0.9)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.savefig('training_history_morandi.png', dpi=300, bbox_inches='tight')
    print("已保存图表: training_history_morandi.png")
    plt.close()

def plot_confusion_matrix_morandi(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    # 自定义渐变色谱 (白 -> 灰绿 -> 灰蓝)
    colors = ["#ffffff", "#e0cdcf", "#9bb0a5", "#7c8b9d", "#4a5a6a"]
    cmap = LinearSegmentedColormap.from_list("MorandiMap", colors, N=256)

    plt.figure(figsize=(16, 14))
    
    ax = sns.heatmap(cm, annot=False, cmap=cmap, fmt='d', 
                     linewidths=0, rasterized=True,
                     cbar_kws={'label': 'Sample Count', 'shrink': 0.8})
    
    ax.set_title('Confusion Matrix (SEI ResNet)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14, labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, labelpad=10)
    
    tick_locator = ticker.MultipleLocator(20)
    ax.xaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_locator(tick_locator)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_morandi.png', dpi=300, bbox_inches='tight')
    print("已保存图表: confusion_matrix_morandi.png")
    plt.close()

def train_model():
    train_data, val_data, X_test, test_df_original = load_data()
    
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = SEResNet1D().to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=True)
    
    print(f"开始训练 SE-ResNet1D (Epochs: {CONFIG['epochs']})...")
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # 用于混淆矩阵的最佳预测缓存
    best_val_targets = []
    best_val_preds = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录数据
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_val_targets = val_targets
            best_val_preds = val_preds
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"训练结束。最佳验证集准确率: {best_acc:.4f}")
    
    plot_history_morandi(history)
    plot_confusion_matrix_morandi(best_val_targets, best_val_preds)
    
    # TTA 推理
    print(f"正在进行 TTA 推理 (每样本 {CONFIG['tta_steps']} 次)...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    final_probs = np.zeros((len(X_test), CONFIG['num_classes']))
    
    for i in range(CONFIG['tta_steps']):
        # 第一次不加噪声，后面几次加噪声
        mode = 'val' if i == 0 else 'test_tta'
        tta_dataset = SEIDataset(X_test, mode=mode)
        tta_loader = DataLoader(tta_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
        probs_list = []
        with torch.no_grad():
            for inputs in tta_loader:
                inputs = inputs.to(CONFIG['device'])
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())
        
        step_probs = np.concatenate(probs_list, axis=0)
        final_probs += step_probs
        print(f"  TTA Step {i+1}/{CONFIG['tta_steps']} 完成")
    
    final_preds = np.argmax(final_probs, axis=1)
            
    if test_df_original.shape[1] > X_test.shape[1] * train_data.features.shape[1]: 
         test_df_original.iloc[:, -1] = final_preds
    else:
         test_df_original['Label'] = final_preds
         
    output_filename = '23009200961.csv'
    test_df_original.to_csv(output_filename, index=False, header=None)
    print(f"TTA 结果已保存至 {output_filename}")

if __name__ == '__main__':
    train_model()
