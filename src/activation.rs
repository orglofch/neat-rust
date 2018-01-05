#[derive(Clone, Copy)]
pub enum ActivationFn {
    Sigmoid,
    Tanh,
    Sin,
    Gauss,
    ReLU,
    SoftPlus,
    Identity,
    Clamped,
    Inv,
    Log,
    Exp,
    Abs,
    Hat,
    Square,
    Cube,
}

impl ActivationFn {
    pub(crate) fn eval(&self, val: f32) -> f32 {
        match *self {
            ActivationFn::Sigmoid => sigmoid_activation(val),
            ActivationFn::Tanh => tanh_activation(val),
            ActivationFn::Sin => sin_activation(val),
            ActivationFn::Gauss => gauss_activation(val),
            ActivationFn::ReLU => relu_activation(val),
            ActivationFn::SoftPlus => softplus_activation(val),
            ActivationFn::Identity => identity_activation(val),
            ActivationFn::Clamped => clamped_activation(val),
            ActivationFn::Inv => inv_activation(val),
            ActivationFn::Log => log_activation(val),
            ActivationFn::Exp => exp_activation(val),
            ActivationFn::Abs => abs_activation(val),
            ActivationFn::Hat => hat_activation(val),
            ActivationFn::Square => square_activation(val),
            ActivationFn::Cube => cube_activation(val),
        }
    }
}

#[inline]
fn sigmoid_activation(val: f32) -> f32 {
    1.0 / (1.0 + (-val).exp())
}

#[inline]
fn tanh_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn sin_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn gauss_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn relu_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn softplus_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn identity_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn clamped_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn inv_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn log_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn exp_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn abs_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn hat_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn square_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn cube_activation(val: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

// TODO(orglofch): Add some more robust goldens.
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_sigmoid_activation() {
        let function = ActivationFn::Sigmoid;

        assert_approx_eq!(function.eval(0.0), 0.5);
        assert_approx_eq!(function.eval(100.0), 1.0);
        assert_approx_eq!(function.eval(-100.0), 0.0);
    }
}
