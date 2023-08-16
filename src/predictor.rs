use crate::ctypes::PD_ConfigCreate;
use crate::tensor::Tensor;
use crate::utils::to_c_str;
use crate::{common::OneDimArrayCstr, ctypes::PD_Bool};
use crate::{config::Config, ctypes::*};
use crate::{
    config::SetConfig,
    ctypes::{
        PD_Predictor, PD_PredictorClone, PD_PredictorDestroy, PD_PredictorGetInputHandle,
        PD_PredictorGetInputNames, PD_PredictorGetInputNum, PD_PredictorGetOutputHandle,
        PD_PredictorGetOutputNames, PD_PredictorGetOutputNum, PD_PredictorRun,
    },
};

/// Paddle Inference 的预测器
#[derive(Debug)]
pub struct Predictor {
    ptr: *mut PD_Predictor,
}

impl Predictor {
    pub fn new(config: Config) -> Predictor {
        let Config {
            model,
            cpu,
            gpu,
            xpu,
            onnx_runtime,
            ir_optimization,
            ir_debug,
            lite,
            memory_optimization,
            optimization_cache_dir,
            disable_fc_padding,
            profile,
            disable_log,
        } = config;

        let config = unsafe { PD_ConfigCreate() };

        model.set_to(config);

        cpu.set_to(config);

        if let Some(g) = gpu {
            g.set_to(config)
        }
        if let Some(x) = xpu {
            x.set_to(config)
        }
        if let Some(o) = onnx_runtime {
            o.set_to(config)
        }

        unsafe { PD_ConfigSwitchIrOptim(config, ir_optimization) };
        unsafe { PD_ConfigSwitchIrDebug(config, ir_debug) };
        // unsafe {
        //     PD_ConfigEnableTensorRtEngine(
        //         config,
        //         1 << 20,
        //         1,
        //         3,
        //         PD_PRECISION_FLOAT32 as i32,
        //         FALSE as i8,
        //         FALSE as i8,
        //     );
        // }
        // TODO: PD_ConfigMkldnnEnabled
        // unsafe {
        //     PD_ConfigSetMkldnnCacheCapacity(config, 10);
        // }

        if let Some(l) = lite {
            l.set_to(config)
        }

        unsafe { PD_ConfigEnableMemoryOptim(config, memory_optimization) };

        if let Some(s) = optimization_cache_dir {
            let (_s, cs) = to_c_str(&s);
            unsafe { PD_ConfigSetOptimCacheDir(config, cs) };
        }

        if disable_fc_padding {
            unsafe { PD_ConfigDisableFCPadding(config) };
        }

        if profile {
            unsafe { PD_ConfigProfileEnabled(config) };
        }

        if disable_log {
            unsafe { PD_ConfigDisableGlogInfo(config) };
        }

        let ptr = unsafe { PD_PredictorCreate(config) };
        Predictor::from_ptr(ptr)
    }

    pub(crate) fn from_ptr(ptr: *mut PD_Predictor) -> Self {
        Self { ptr }
    }
}

impl Predictor {
    /// 获取输入 Tensor 名称
    pub fn input_names(&self) -> OneDimArrayCstr {
        let ptr = unsafe { PD_PredictorGetInputNames(self.ptr) };
        OneDimArrayCstr::from_ptr(ptr)
    }

    /// 获取输入 Tensor 数量
    pub fn input_num(&self) -> usize {
        unsafe { PD_PredictorGetInputNum(self.ptr) }
    }

    /// 根据名称获取输入 Tensor
    ///
    /// **注意:** 如果输入名称中包含字符`\0`，则只会将`\0`之前的字符作为输入
    pub fn input(&self, name: &str) -> Tensor {
        let (_n, name) = to_c_str(name);
        let ptr = unsafe { PD_PredictorGetInputHandle(self.ptr, name) };
        Tensor::from_ptr(ptr)
    }

    /// 获取输出 Tensor 名称
    pub fn output_names(&self) -> OneDimArrayCstr {
        let ptr = unsafe { PD_PredictorGetOutputNames(self.ptr) };
        OneDimArrayCstr::from_ptr(ptr)
    }

    /// 获取输出 Tensor 数量
    pub fn output_num(&self) -> usize {
        unsafe { PD_PredictorGetOutputNum(self.ptr) }
    }

    /// 根据名称获取输出 Tensor
    ///
    /// **注意:** 如果输入名称中包含字符`\0`，则只会将`\0`之前的字符作为输入
    pub fn output(&self, name: &str) -> Tensor {
        let (_n, name) = to_c_str(name);
        let ptr = unsafe { PD_PredictorGetOutputHandle(self.ptr, name) };
        Tensor::from_ptr(ptr)
    }
}

impl Predictor {
    /// 执行模型预测，**需要在设置输入Tensor数据后调用**
    pub fn run(&self) -> PD_Bool {
        unsafe { PD_PredictorRun(self.ptr) }
    }

    pub fn clear_intermediate_tensor(&self) {
        // unsafe { PD_PredictorClearIntermediateTensor(self.ptr) };
    }
}

impl Clone for Predictor {
    fn clone(&self) -> Self {
        let ptr = unsafe { PD_PredictorClone(self.ptr) };
        Self { ptr }
    }
}

impl Drop for Predictor {
    fn drop(&mut self) {
        unsafe { PD_PredictorDestroy(self.ptr) };
    }
}
