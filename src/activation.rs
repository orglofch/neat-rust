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
    Sinc,
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
            ActivationFn::Sinc => sinc_activation(val),
        }
    }
}

// TODO(orglofch): Python neat skews the input numbers.
// TODO(orglofch): Consider others https://en.wikipedia.org/wiki/Activation_function

#[inline]
fn sigmoid_activation(val: f32) -> f32 {
    1.0 / (1.0 + (-val).exp())
}

#[inline]
fn tanh_activation(val: f32) -> f32 {
    val.tanh()
}

#[inline]
fn sin_activation(val: f32) -> f32 {
    val.sin()
}

#[inline]
fn gauss_activation(val: f32) -> f32 {
    (-1.0 * val * val).exp()
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

#[inline]
fn sinc_activation(val: f32) -> f32 {
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

    #[test]
    pub fn test_tanh_activation() {
        let function = ActivationFn::Tanh;

        assert_approx_eq!(function.eval(0.0), (0.0_f32).tanh());
        assert_approx_eq!(function.eval(1.0), (1.0_f32).tanh());
        assert_approx_eq!(function.eval(0.5), (0.5_f32).tanh());
    }

    #[test]
    pub fn test_sin_activation() {
        let function = ActivationFn::Sin;

        assert_approx_eq!(function.eval(0.0), (0.0_f32).sin());
        assert_approx_eq!(function.eval(1.0), (1.0_f32).sin());
        assert_approx_eq!(function.eval(0.5), (0.5_f32).sin());
    }

    #[test]
    pub fn test_gauss_activation() {
        let function = ActivationFn::Gauss;

        assert_approx_eq!(function.eval(0.0), 1.0);
        assert_approx_eq!(function.eval(100.0), 0.0);
        assert_approx_eq!(function.eval(-100.0), 0.0);
    }
}
