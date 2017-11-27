#[derive(Clone)]
pub enum ActivationFn {
    Sigmoid,
}

impl ActivationFn {
    pub(crate) fn eval(&self, val: f32) -> f32 {
        match self {
            &ActivationFn::Sigmoid => sigmoid_activation(val),
        }
    }
}

fn sigmoid_activation(val: f32) -> f32 {
    1.0 / (1.0 + (-val).exp())
}
