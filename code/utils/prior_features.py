import numpy as np
from scipy.signal import hilbert
from scipy.stats import entropy, skew
from typing import Optional


# === 先验特征工具 ===
# 将时间域/频域信号转换为统计指标（最大值、峰值、均方根、峰度等），供数据模块拼接进模型输入。
_EPS = 1e-8


def _ensure_2d(data: Optional[np.ndarray]) -> Optional[np.ndarray]: # ->是Python中用于标注函数返回值的语法，Optional表示该类型是可选的，None或者是一个numpy数组

    """
    确保输入的数组是二维的

    参数:
        data: 输入的数据，可以是None或numpy数组

    返回:
        如果输入是None，则返回None
        如果输入是一维数组，则将其转换为二维数组（1行，多列）
        如果输入已经是二维数组，则直接返回

    注意:
        此函数不会修改原始数据，而是返回一个新的数组
    """
    if data is None:  # 如果输入为None，直接返回None
        return None
    arr = np.asarray(data)  # 将输入转换为numpy数组
    if arr.ndim == 1:  # 检查是否为一维数组
        arr = arr.reshape(1, -1)  # 将一维数组转换为二维数组（1行，多列）
    return arr  # 返回处理后的二维数组


def compute_prior_indicators(time_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """从振动样本中计算15个先验指标 (p1~p15)。
    指标解释见函数内部各变量旁的注释。
    返回值：ndarray，形状 (n_samples, 15)，按列顺序为：
      1 max_val (最大值)、2 min_val (最小值)、3 peak (绝对峰值)、4 range (极差)、
      5 abs_mean (整流平均)、6 sra (平方根幅值指标)、7 mean_square (均方)、
      8 std (样本标准差)、9 rms (均方根)、10 kurtosis (峰度)、
      11 skew_val (偏度)、12 crest_factor (峰值因子)、13 impulse_factor (脉冲因子)、
      14 shape_factor (波形因子)、15 clearance_factor (裕度因子)
    """
    # 针对每个样本计算 15 项时域指标，可复用在 time/freq 两种来源上。
    data = _ensure_2d(time_data)
    if data is None:
        return None
    # 基本幅值统计
    mean = data.mean(axis=1, keepdims=True)    # 均值：基线/偏移检测（平均值）
    abs_data = np.abs(data)                    # 绝对值：用于计算整流平均等幅值度量
    max_val = data.max(axis=1, keepdims=True)  # 最大值：瞬时最大幅度，敏感冲击或极端事件
    min_val = data.min(axis=1, keepdims=True)  # 最小值：瞬时最小幅度
    peak = abs_data.max(axis=1, keepdims=True) # 绝对峰值：峰值强度指示，常用于冲击检测
    range = max_val - min_val                  # 极差：幅值范围，反映剧烈波动或不稳定性
    abs_mean = abs_data.mean(axis=1, keepdims=True) # 整流平均：平均绝对幅度，常与能量相关
    # SRA（平方根幅值指标）：对短时冲击、磨损或间隙变化敏感
    sra = np.square(np.mean(np.sqrt(abs_data + _EPS), axis=1, keepdims=True))
    mean_square = np.mean(np.square(data), axis=1, keepdims=True) # 均方：与信号能量/功率直接相关
    variance = np.var(data, axis=1, keepdims=True, ddof=1)  # 样本方差（用于计算样本标准差）
    std = np.sqrt(variance + _EPS)                 # 样本标准差：幅值波动/抖动强度
    rms = np.sqrt(np.mean(np.square(data), axis=1, keepdims=True)) # 均方根（RMS）：有效值，常用能量度量
    std_population = np.sqrt(np.var(data, axis=1, keepdims=True, ddof=0)) # 总体标准差（用于峰度计算）
    kurtosis = np.mean(np.power((data - mean) / (std_population + _EPS), 4), axis=1, keepdims=True) # 峰度：衡量尖锐/突发冲击特性
    
    # === 新增：无量纲指标与高阶统计量（工程性比值，常用于冲击/磨损检测） ===
    # 11. 偏度 (Skewness): 描述分布的不对称性，可能指示偏移或方向性脉冲
    skew_val = skew(data, axis=1).reshape(-1, 1)
    # 12. 峰值因子 (Crest Factor): 峰值 / RMS，峰值相对于能量的放大，敏感冲击
    crest_factor = peak / (rms + _EPS)
    # 13. 脉冲因子 (Impulse Factor): 峰值 / 整流平均，对短时脉冲/冲击敏感
    impulse_factor = peak / (abs_mean + _EPS)
    # 14. 波形因子 (Shape Factor): RMS / 整流平均，表征波形的平滑/粗糙程度
    shape_factor = rms / (abs_mean + _EPS)
    # 15. 裕度因子 (Clearance Factor): 峰值 / SRA，对磨损、间隙或局部强冲击敏感
    clearance_factor = peak / (sra + _EPS) 

    features = [
        max_val,        # 1 最大值（瞬时最大幅值，冲击敏感）
        min_val,        # 2 最小值（瞬时最小幅值）
        peak,           # 3 绝对峰值（冲击强度指示）
        range,          # 4 极差（幅值范围，反映波动）
        abs_mean,       # 5 整流平均（平均绝对幅度，能量相关）
        sra,            # 6 SRA（平方根幅值），对小冲击/磨损敏感
        mean_square,    # 7 均方（能量）
        std,            # 8 样本标准差（幅值波动）
        rms,            # 9 RMS（有效值/能量密度）
        kurtosis,       # 10 峰度（冲击尖峰程度）
        skew_val,       # 11 偏度（分布不对称性）
        crest_factor,   # 12 峰值因子（峰值/RMS，冲击指示）
        impulse_factor, # 13 脉冲因子（峰值/整流平均，冲击指示）
        shape_factor,   # 14 波形因子（RMS/整流平均，波形形状）
        clearance_factor# 15 裕度因子（峰值/SRA，对磨损/间隙敏感）
    ]
    return np.concatenate(features, axis=1) #将计算得到的特征沿列方向拼接起来，并返回最终的特征矩阵


def compute_spectral_entropy(freq_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """从频域数据计算谱熵"""
    data = _ensure_2d(freq_data)
    if data is None:
        return None
    # 归一化以获得概率分布
    # 添加小常数以避免除以零
    psd = data + _EPS
    psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
    # 使用scipy的entropy函数计算熵
    se = entropy(psd_norm, axis=1)
    return se.reshape(-1, 1) # 返回1列谱熵（Spectral Entropy），值越高表示频谱越均匀/复杂


def compute_band_energy_stats(freq_data: Optional[np.ndarray], num_bands: int = 3) -> Optional[np.ndarray]:
    """计算频带能量比 (Low/Mid/High Frequency Energy Ratios)"""
    data = _ensure_2d(freq_data)
    if data is None:
        return None
    
    # 将频谱均分为 num_bands 段
    n_points = data.shape[1]
    band_width = n_points // num_bands
    
    energy_ratios = []
    total_energy = np.sum(np.square(data), axis=1, keepdims=True) + _EPS
    
    for i in range(num_bands):
        start = i * band_width
        end = (i + 1) * band_width if i < num_bands - 1 else n_points
        
        # 计算该频带的能量
        band_energy = np.sum(np.square(data[:, start:end]), axis=1, keepdims=True)
        # 计算能量占比
        ratio = band_energy / total_energy
        energy_ratios.append(ratio)
        
    return np.concatenate(energy_ratios, axis=1) # 返回 num_bands 列的频带能量占比（每列为对应频段的能量比）


def compute_cepstrum_stats(freq_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """计算倒频谱特征 (Cepstrum Features)，用于检测边频带和谐波"""
    data = _ensure_2d(freq_data)
    if data is None:
        return None
    
    # 倒频谱计算: IFFT(log(abs(FFT)))
    # 注意：输入的 freq_data 已经是 abs(FFT) 了
    log_spectrum = np.log(data + _EPS)
    cepstrum = np.abs(np.fft.ifft(log_spectrum, axis=1))
    
    # 忽略倒频谱原本点（Quefrency=0附近），通常包含直流分量
    # 取第 5 个点之后的区域进行统计
    valid_cepstrum = cepstrum[:, 5:]
    
    # 提取倒频谱的统计特征
    cep_max = np.max(valid_cepstrum, axis=1, keepdims=True)
    cep_mean = np.mean(valid_cepstrum, axis=1, keepdims=True)
    
    return np.concatenate([cep_max, cep_mean], axis=1) # 返回2列倒频谱特征：cep_max（倒频谱最大值）、cep_mean（倒频谱均值），用于检测谐波或边频带特征


def compute_envelope_stats(time_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """从希尔伯特包络谱计算统计量，包络统计量处理的是经过希尔伯特变换后的信号包络，与上述的先验特征不一样"""
    data = _ensure_2d(time_data)
    if data is None:
        return None
    # 希尔伯特变换
    # 确保输入是连续的，避免内存布局问题
    data = np.ascontiguousarray(data)
    analytic_signal = hilbert(data, axis=1) #hilbert函数用于计算信号的希尔伯特变换，返回复数信号，其中实部是原始信号，虚部是原始信号的解析信号
    amplitude_envelope = np.abs(analytic_signal) #计算解析信号的模，得到包络
    
    # 计算包络的统计量（类似于时域统计量）
    # 包络的均值、标准差、最大值、峰度
    mean_env = np.mean(amplitude_envelope, axis=1, keepdims=True)
    std_env = np.std(amplitude_envelope, axis=1, keepdims=True, ddof=1)
    max_env = np.max(amplitude_envelope, axis=1, keepdims=True)
    
    std_pop = np.std(amplitude_envelope, axis=1, keepdims=True, ddof=0)
    kurtosis_env = np.mean(np.power((amplitude_envelope - mean_env) / (std_pop + _EPS), 4), axis=1, keepdims=True)
    
    return np.concatenate([mean_env, std_env, max_env, kurtosis_env], axis=1) # 返回4列 envelope 特征：mean_env, std_env, max_env, kurtosis_env（包络均值、标准差、最大值、峰度）


def compute_spectral_moments(freq_data: Optional[np.ndarray], freq_axis: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    计算一阶谱矩特征 (Spectral Moments)
    包括：谱质心 (Centroid)、谱方差 (Spread)、谱偏度 (Skewness)、谱峰度 (Kurtosis)
    """
    data = _ensure_2d(freq_data)
    if data is None:
        return None
    
    # 如果没有提供频率轴，假设为线性分布 0, 1, 2...
    if freq_axis is None:
        freq_axis = np.arange(data.shape[1]).reshape(1, -1)
    else:
        freq_axis = _ensure_2d(freq_axis)

    # 归一化功率谱 P(f)
    psd = data + _EPS
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_norm = psd / psd_sum

    # 1. 谱质心 (Spectral Centroid) - 一阶矩
    # 反映频谱能量集中的中心频率
    centroid = np.sum(freq_axis * psd_norm, axis=1, keepdims=True)

    # 2. 谱方差 (Spectral Spread) - 二阶中心矩
    # 反映频谱的带宽或离散程度
    variance = np.sum(((freq_axis - centroid) ** 2) * psd_norm, axis=1, keepdims=True)
    spread = np.sqrt(variance)

    # 3. 谱偏度 (Spectral Skewness) - 三阶中心矩
    # 反映频谱分布的不对称性
    skewness = np.sum(((freq_axis - centroid) ** 3) * psd_norm, axis=1, keepdims=True) / (spread ** 3 + _EPS)

    # 4. 谱峰度 (Spectral Kurtosis) - 四阶中心矩
    # 反映频谱分布的尖锐程度
    kurtosis = np.sum(((freq_axis - centroid) ** 4) * psd_norm, axis=1, keepdims=True) / (spread ** 4 + _EPS)

    return np.concatenate([centroid, spread, skewness, kurtosis], axis=1) # 返回4列谱矩特征：centroid（谱质心）、spread（谱带宽/扩展）、skewness（谱偏度）、kurtosis（谱峰度）


def compute_bispectrum_stats(time_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    计算二阶谱（双谱）的对角切片统计特征
    双谱 B(f1, f2) 在 f1=f2 处的切片，反映了信号的非高斯性和非线性耦合。
    """
    data = _ensure_2d(time_data)
    if data is None:
        return None
    
    N = data.shape[1]
    # 计算 FFT
    fft_val = np.fft.fft(data, axis=1)
    
    # 计算双谱对角切片: B(f, f) = E[X(f) * X(f) * conj(X(2f))]
    # 注意：为了简化计算和保持维度一致，我们只取前半部分频率
    half_N = N // 2
    X_f = fft_val[:, :half_N]
    X_2f = fft_val[:, :half_N:2] # 降采样以匹配 2f 频率
    
    # 截断 X_f 以匹配 X_2f 的长度 (因为 X_2f 长度减半了)
    min_len = min(X_f.shape[1], X_2f.shape[1])
    X_f = X_f[:, :min_len]
    X_2f = X_2f[:, :min_len] # 实际上 X_2f 已经是短的那个了
    
    # 计算对角双谱
    # 确保数组是连续的，避免负步幅问题
    X_f = np.ascontiguousarray(X_f)
    X_2f = np.ascontiguousarray(X_2f)
    bispectrum_diag = np.abs(X_f * X_f * np.conjugate(X_2f))
    
    # 提取统计特征
    bi_mean = np.mean(bispectrum_diag, axis=1, keepdims=True)
    bi_std = np.std(bispectrum_diag, axis=1, keepdims=True)
    bi_max = np.max(bispectrum_diag, axis=1, keepdims=True)
    # 双谱偏度 (Bispectral Skewness) - 归一化
    bi_skew = skew(bispectrum_diag, axis=1).reshape(-1, 1)
    
    return np.concatenate([bi_mean, bi_std, bi_max, bi_skew], axis=1) # 返回4列双谱特征：均值、标准差、最大值、偏度（对非线性/非高斯性敏感）


def build_prior_features(time_data=None, freq_data=None, source: str = "time"): #根据输入的时间域数据(time_data)和/或频域数据(freq_data)计算先验特征指标
                                                                                #默认是time，可以通过source参数选择使用时域数据("time")、频域数据("fft")或两者("both")
    # 根据 source 选择使用哪种信号计算 prior，最后把所有指标拼接在一起返回。
    """根据配置的源返回工程化的先验指标"""
    source = (source or "time").lower() #将source参数转换为小写
    stats = []
    if source in ("time", "both"): #如果source参数为"time"或"both"，则计算时域数据的先验特征指标
        feats = compute_prior_indicators(time_data)
        if feats is not None:
            stats.append(feats) #将计算得到的时域数据的先验特征指标添加到stats列表中
        
        # 为时域添加包络统计量，许多机械故障(如轴承内外圈故障、齿轮点蚀)会产生幅值调制，包络分析能有效提取这些特征
        env_feats = compute_envelope_stats(time_data)
        if env_feats is not None:
            stats.append(env_feats)
        
        # === 新增：二阶谱（双谱）统计特征 ===
        # 对非线性故障（如裂纹呼吸效应）非常敏感
        bispec_feats = compute_bispectrum_stats(time_data)
        if bispec_feats is not None:
            stats.append(bispec_feats)

    if source in ("fft", "both"): #如果source参数为"fft"或"both"，则计算频域数据的先验特征指标
        feats = compute_prior_indicators(freq_data)
        if feats is not None:
            stats.append(feats) #将计算得到的频域数据的先验特征指标添加到stats列表中
        
        # 为频域添加谱熵，频谱熵越高，表示信号频谱分布越均匀，信号越复杂；出现故障时频谱会变得复杂，熵值增加
        se_feats = compute_spectral_entropy(freq_data)
        if se_feats is not None:
            stats.append(se_feats)

        # === 新增：一阶谱矩特征 ===
        # 描述频谱的形状分布
        spec_moments = compute_spectral_moments(freq_data)
        if spec_moments is not None:
            stats.append(spec_moments)

        # === 新增：频带能量比 ===
        # 类似于 H/L 能量比，将频谱分为低、中、高三段，计算能量占比
        band_feats = compute_band_energy_stats(freq_data, num_bands=3)
        if band_feats is not None:
            stats.append(band_feats)
            
        # === 新增：倒频谱特征 ===
        # 用于检测周期性的谐波成分（如齿轮啮合频率的边频带）
        cep_feats = compute_cepstrum_stats(freq_data)
        if cep_feats is not None:
            stats.append(cep_feats)

    if not stats: #如果stats列表为空，则返回None
        return None
    result = np.concatenate(stats, axis=1) #将stats列表中的所有先验特征指标沿列方向拼接起来，并返回最终的特征矩阵,相当于在每行数据的第1026个采样点后面添加0或10+4+1或20+5*2个特征值(包括频域时域)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) #将特征矩阵中的NaN值替换为0.0，正无穷大值替换为0.0，负无穷大值替换为0.0
