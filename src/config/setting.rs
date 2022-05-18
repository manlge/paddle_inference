use crate::common::PrecisionType;
use crate::config::SetConfig;
use crate::ctypes::{
    PD_Config, PD_ConfigEnableCudnn, PD_ConfigEnableGpuMultiStream, PD_ConfigEnableMkldnnBfloat16,
    PD_ConfigEnableONNXRuntime, PD_ConfigEnableORTOptimization, PD_ConfigEnableTensorRtDla,
    PD_ConfigEnableTensorRtEngine, PD_ConfigEnableTensorRtOSS, PD_ConfigEnableUseGpu,
    PD_ConfigEnableXpu, PD_ConfigSetBfloat16Op, PD_ConfigSetCpuMathLibraryNumThreads,
    PD_ConfigSetMkldnnCacheCapacity, PD_ConfigSetMkldnnOp, PD_ConfigSetTrtDynamicShapeInfo,
};
use crate::utils::to_c_str;
use std::ptr::null;

#[derive(Debug, Default)]
pub struct Cpu {
    /// 设置 CPU Blas 库计算线程数
    pub threads: Option<i32>,
    ///  MKLDNN 设置
    pub mkldnn: Option<Mkldnn>,
}

#[derive(Debug)]
pub struct Gpu {
    /// 初始化分配的gpu显存，以MB为单位
    pub memory_pool_init_size_mb: u64,
    /// 设备id
    pub device_id: i32,
    /// 开启线程流，目前的行为是为每一个线程绑定一个流，在将来该行为可能改变
    pub enable_multi_stream: bool,
    /// 启用 CUDNN 进行预测加速
    pub enable_cudnn: bool,
    /// TensorRT 设置
    pub enable_tensor_rt: Option<TensorRT>,
}

#[derive(Debug)]
pub struct Xpu {
    /// l3 cache 分配的显存大小。单位为 mb
    pub l3_workspace_size_mb: i32,
    /// 是否独占申请的 l3 cache。如果为false，分配的多个 l3 cache 可以被多个模型共享
    ///
    /// Whether the allocated L3 cache can be locked. If false, it means that the L3 cache is
    /// not locked, and the allocated L3 cache can be shared by multiple models, and multiple
    /// models sharing the L3 cache will be executed sequentially on the card.
    pub locked: bool,
    /// Whether to autotune the conv operator in the model. If true, when the conv operator of
    /// a certain dimension is executed for the first time, it will automatically search for a
    /// better algorithm to improve the performance of subsequent conv operators of the same
    /// dimension.
    pub autorune: bool,
    /// Specify the path of the autotune file. If autotune_file is specified, the algorithm
    /// specified in the file will be used and autotune will not be performed again.
    pub autotune_file: Option<String>,
    /// Calculation accuracy of multi_encoder
    pub precision: String,
    /// Is the input of multi_encoder variable length
    pub adaptive_seqlen: bool,
}

#[derive(Debug)]
pub struct ONNXRuntime {
    /// 启用 ONNXRuntime 预测时开启优化
    pub enable_optimization: bool,
}

#[derive(Debug, Default)]
pub struct Mkldnn {
    /// 设置 MKLDNN 针对不同输入 shape 的 cache 容量大小
    pub cache_size: Option<i32>,
    /// 指定使用 MKLDNN 加速的 OP 列表
    pub op: Option<Vec<String>>,
    /// 启用 MKLDNN BFLOAT16 并指定使用 MKLDNN BFLOAT16 加速的 OP 列表
    pub op_f16: Option<Vec<String>>,
}

/// TensorRT 设置
///
/// **注意：**
///
/// 1. 启用 TensorRT 的前提为已经启用 GPU，否则启用 TensorRT 无法生效
/// 2. 对存在LoD信息的模型，如Bert, Ernie等NLP模型，必须使用动态 Shape
/// 3. 启用 TensorRT OSS 可以支持更多 plugin，详细参考 [TensorRT OSS](https://news.developer.nvidia.com/nvidia-open-sources-parsers-and-plugins-in-tensorrt/)
///
/// 更多 TensorRT 详细信息，请参考[使用Paddle-TensorRT库预测](https://paddleinference.paddlepaddle.org.cn/optimize/paddle_trt.html)。
#[derive(Debug)]
pub struct TensorRT {
    /// 指定 TensorRT 使用的工作空间大小
    pub workspace_size: i32,
    /// 设置最大的 batch 大小，运行时 batch 大小不得超过此限定值
    pub max_batch_size: i32,
    /// Paddle-TRT 是以子图的形式运行，为了避免性能损失，当子图内部节点个数大于 min_subgraph_size 的时候，才会使用
    /// Paddle-TRT 运行
    pub min_subgraph_size: i32,
    /// 指定使用 TRT 的精度，支持 FP32(kFloat32)，FP16(kHalf)，Int8(kInt8)
    pub precision_type: PrecisionType,
    /// 若指定为 TRUE，在初次运行程序的时候会将 TRT 的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而
    /// 不需要重新生成
    pub use_static: bool,
    /// 若要运行 Paddle-TRT INT8 离线量化校准，需要将此选项设置为 TRUE
    pub use_calib_mode: bool,
    /// TensorRT 的动态 Shape. 长度为`0`时不设置
    pub dynamic_shape_info: Vec<DynamicShapeInfo>,
    /// 设置 TensorRT 的 plugin 不在 fp16 精度下运行
    ///
    /// **仅在[`Self::dynamic_shape_info`]字段长度大于0时有效**
    pub disable_plugin_fp16: bool,
    /// 启用 TensorRT OSS 进行预测加速
    pub enable_oss: bool,
    /// 启用TensorRT DLA进行预测加速. 值为DLA设备的id，可选0，1，...，DLA设备总数 - 1
    pub dla_core: Option<i32>,
}

/// TensorRT 的动态 Shape 信息
///
/// **注意：** DynamicShapeInfo 中所有shape的大小必须相同
#[derive(Debug)]
pub struct DynamicShapeInfo {
    /// Tensor 名称
    pub name: String,
    /// Tensor 对应的最小 shape
    pub min_shape: Vec<i32>,
    /// Tensor 对应的最大 shape
    pub max_shape: Vec<i32>,
    /// Tensor 对应的最优 shape
    pub optim_shape: Vec<i32>,
}

impl DynamicShapeInfo {
    #[inline]
    pub fn check_and_get_shape_size(&self) -> usize {
        assert!(
            self.min_shape.len() == self.max_shape.len()
                && self.min_shape.len() == self.optim_shape.len(),
            "DynamicShapeInfo 中所有shape的大小必须相同"
        );
        self.min_shape.len()
    }
}

impl SetConfig for Cpu {
    fn set_to(self, config: *mut PD_Config) {
        let Cpu { threads, mkldnn } = self;
        if let Some(t) = threads {
            if t > 0 {
                unsafe { PD_ConfigSetCpuMathLibraryNumThreads(config, t) };
            }
        }
        if let Some(Mkldnn {
            cache_size,
            op,
            op_f16,
        }) = mkldnn
        {
            if let Some(cache) = cache_size {
                if cache > 0 {
                    unsafe { PD_ConfigSetMkldnnCacheCapacity(config, cache) };
                }
            }
            if let Some(op) = op {
                let (_l, mut r): (Vec<_>, Vec<_>) = op.iter().map(|s| to_c_str(s)).unzip();
                let size = r.len();
                unsafe {
                    PD_ConfigSetMkldnnOp(config, size, r.as_mut_ptr());
                }
            }
            if let Some(op) = op_f16 {
                unsafe { PD_ConfigEnableMkldnnBfloat16(config) };
                let (_l, mut r): (Vec<_>, Vec<_>) = op.iter().map(|s| to_c_str(s)).unzip();
                let size = r.len();
                unsafe {
                    PD_ConfigSetBfloat16Op(config, size, r.as_mut_ptr());
                }
            }
        }
    }
}

impl SetConfig for Gpu {
    fn set_to(self, config: *mut PD_Config) {
        let Gpu {
            memory_pool_init_size_mb,
            device_id,
            enable_multi_stream,
            enable_cudnn,
            enable_tensor_rt,
        } = self;

        unsafe {
            PD_ConfigEnableUseGpu(config, memory_pool_init_size_mb, device_id);
        }
        if enable_multi_stream {
            unsafe { PD_ConfigEnableGpuMultiStream(config) };
        }
        if enable_cudnn {
            unsafe { PD_ConfigEnableCudnn(config) };
        }
        if let Some(TensorRT {
            workspace_size,
            max_batch_size,
            min_subgraph_size,
            precision_type,
            use_static,
            use_calib_mode,
            dynamic_shape_info,
            disable_plugin_fp16,
            enable_oss,
            dla_core,
        }) = enable_tensor_rt
        {
            unsafe {
                PD_ConfigEnableTensorRtEngine(
                    config,
                    workspace_size,
                    max_batch_size,
                    min_subgraph_size,
                    precision_type,
                    use_static,
                    use_calib_mode,
                );
            }

            if !dynamic_shape_info.is_empty() {
                let tensor_num = dynamic_shape_info.len();
                let mut tensor_name = vec![];
                let mut tensor_name_cs = vec![];
                let mut shapes_num = vec![];
                let mut min_shapes = vec![];
                let mut max_shapes = vec![];
                let mut optim_shapes = vec![];
                for info @ DynamicShapeInfo {
                    name,
                    min_shape,
                    max_shape,
                    optim_shape,
                } in &dynamic_shape_info
                {
                    shapes_num.push(info.check_and_get_shape_size());

                    let (cn, n) = to_c_str(name);
                    tensor_name.push(n);
                    tensor_name_cs.push(cn);

                    min_shapes.push(min_shape.as_ptr() as *mut i32);
                    max_shapes.push(max_shape.as_ptr() as *mut i32);
                    optim_shapes.push(optim_shape.as_ptr() as *mut i32);
                }

                unsafe {
                    PD_ConfigSetTrtDynamicShapeInfo(
                        config,
                        tensor_num,
                        tensor_name.as_mut_ptr(),
                        shapes_num.as_mut_ptr(),
                        min_shapes.as_mut_ptr(),
                        max_shapes.as_mut_ptr(),
                        optim_shapes.as_mut_ptr(),
                        disable_plugin_fp16,
                    );
                }
            }

            if enable_oss {
                unsafe { PD_ConfigEnableTensorRtOSS(config) };
            }

            if let Some(dla_core) = dla_core {
                unsafe { PD_ConfigEnableTensorRtDla(config, dla_core) };
            }
        }
    }
}

impl SetConfig for Xpu {
    fn set_to(self, config: *mut PD_Config) {
        let Xpu {
            l3_workspace_size_mb,
            locked,
            autorune,
            autotune_file,
            precision,
            adaptive_seqlen,
        } = self;
        let (_a, af) = autotune_file
            .as_ref()
            .map(|s| to_c_str(s))
            .unwrap_or_else(|| (None, null()));
        let (_p, p) = to_c_str(&precision);

        unsafe {
            PD_ConfigEnableXpu(
                config,
                l3_workspace_size_mb,
                locked,
                autorune,
                af,
                p,
                adaptive_seqlen,
            );
        }
    }
}

impl SetConfig for ONNXRuntime {
    fn set_to(self, config: *mut PD_Config) {
        unsafe { PD_ConfigEnableONNXRuntime(config) };
        if self.enable_optimization {
            unsafe { PD_ConfigEnableORTOptimization(config) };
        }
    }
}
