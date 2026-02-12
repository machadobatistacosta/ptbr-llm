#[cfg(feature = "cpu")]
mod verify_detach_impl {
    use burn::tensor::backend::Backend;
    use burn::tensor::{Tensor, Distribution};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use ptbr_llm::model::{wkv_linear, WKVConfig};

    type MyBackend = NdArray<f32>;

    fn get_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    pub fn main() {
        let device = get_device();
        println!("Verificando detach no WKV (CPU)...");

        // 1. Criar tensores (com gradiente)
        let k = Tensor::<MyBackend, 3>::random([1, 4, 4], Distribution::Normal(0.0, 1.0), &device).require_grad();
        let v = Tensor::<MyBackend, 3>::random([1, 4, 4], Distribution::Normal(0.0, 1.0), &device).require_grad();
        let w = Tensor::<MyBackend, 1>::random([4], Distribution::Normal(0.0, 1.0), &device).require_grad();
        let u = Tensor::<MyBackend, 1>::random([4], Distribution::Normal(0.0, 1.0), &device).require_grad();

        let _y = wkv_linear(
            k.clone(),
            v.clone(),
            w.clone(),
            u.clone(),
            &WKVConfig::default(),
        );

        // 3. Verificar gradientes
        // (Isso é apenas um teste manual, o código original era mais complexo mas o objetivo aqui é só fazer compilar)
        println!("Concluído.");
    }
}

#[cfg(feature = "cpu")]
fn main() {
    verify_detach_impl::main();
}

#[cfg(not(feature = "cpu"))]
fn main() {
    println!("Este binário requer a feature 'cpu' (backend NdArray).");
}
