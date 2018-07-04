const EPSILON: f32 = 1e-7;

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
    Ln,
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
            ActivationFn::Ln => ln_activation(val),
            ActivationFn::Exp => exp_activation(val),
            ActivationFn::Abs => abs_activation(val),
            ActivationFn::Hat => hat_activation(val),
            ActivationFn::Square => square_activation(val),
            ActivationFn::Cube => cube_activation(val),
            ActivationFn::Sinc => sinc_activation(val),
        }
    }
}

// TODO(orglofch): Skew the inputs to prevent infinite values (like in python neat).
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
fn relu_activation(_: f32) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn softplus_activation(val: f32) -> f32 {
    (1.0 + val.exp()).ln()
}

#[inline]
fn identity_activation(val: f32) -> f32 {
    val
}

#[inline]
fn clamped_activation(val: f32) -> f32 {
    val.max(-1.0).min(1.0)
}

#[inline]
fn inv_activation(val: f32) -> f32 {
    let inv = 1.0 / val;
    if inv.is_infinite() {
        0.0
    } else {
        inv
    }
}

#[inline]
fn ln_activation(val: f32) -> f32 {
    val.max(EPSILON).ln()
}

#[inline]
fn exp_activation(val: f32) -> f32 {
    val.exp()
}

#[inline]
fn abs_activation(val: f32) -> f32 {
    val.abs()
}

#[inline]
fn hat_activation(val: f32) -> f32 {
    (1.0 - val.abs()).max(0.0)
}

#[inline]
fn square_activation(val: f32) -> f32 {
    val * val
}

#[inline]
fn cube_activation(val: f32) -> f32 {
    val * val * val
}

#[inline]
fn sinc_activation(_: f32) -> f32 {
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

    #[test]
    pub fn test_softplus_activation() {
        let function = ActivationFn::SoftPlus;

        assert_approx_eq!(function.eval(0.0), (2.0_f32).ln());
        assert_approx_eq!(function.eval(-100.0), 0.0);
    }

    #[test]
    pub fn test_identity_activation() {
        let function = ActivationFn::Identity;

        assert_approx_eq!(function.eval(0.0), 0.0);
        assert_approx_eq!(function.eval(10.0), 10.0);
        assert_approx_eq!(function.eval(-10.0), -10.0);
    }

    #[test]
    pub fn test_clamped_activation() {
        let function = ActivationFn::Clamped;

        assert_approx_eq!(function.eval(0.0), 0.0);
        assert_approx_eq!(function.eval(2.0), 1.0);
        assert_approx_eq!(function.eval(-2.0), -1.0);
    }

    #[test]
    pub fn test_inv_activation() {
        let function = ActivationFn::Inv;

        assert_approx_eq!(function.eval(0.0), 0.0);
        assert_approx_eq!(function.eval(1.0), 1.0);
        assert_approx_eq!(function.eval(2.0), 0.5);
        assert_approx_eq!(function.eval(-4.0), -0.25);
    }

    #[test]
    pub fn test_ln_activation() {
        let function = ActivationFn::Ln;

        assert_approx_eq!(function.eval(0.0), EPSILON.ln());
        assert_approx_eq!(function.eval(2.0), (2.0_f32).ln());
        assert_approx_eq!(function.eval(-2.0), EPSILON.ln());
    }

    #[test]
    pub fn test_exp_activation() {
        let function = ActivationFn::Exp;

        assert_approx_eq!(function.eval(0.0), (0.0_f32).exp());
        assert_approx_eq!(function.eval(2.0), (2.0_f32).exp());
        assert_approx_eq!(function.eval(-2.0), (-2.0_f32).exp());
    }

    #[test]
    pub fn test_abs_activation() {
        let function = ActivationFn::Abs;

        assert_approx_eq!(function.eval(0.0), 0.0);
        assert_approx_eq!(function.eval(10.0), 10.0);
        assert_approx_eq!(function.eval(-10.0), 10.0);
    }

    #[test]
    pub fn test_hat_activation() {
        let function = ActivationFn::Hat;

        assert_approx_eq!(function.eval(0.0), 1.0);
        assert_approx_eq!(function.eval(1.0), 0.0);
        assert_approx_eq!(function.eval(-1.0), 0.0);
        assert_approx_eq!(function.eval(0.5), 0.5);
        assert_approx_eq!(function.eval(-0.5), 0.5);
    }

    #[test]
    pub fn test_square_activation() {
        let function = ActivationFn::Square;

        assert_approx_eq!(function.eval(0.0), 0.0);
        assert_approx_eq!(function.eval(2.0), 4.0);
        assert_approx_eq!(function.eval(-2.0), 4.0);
    }

    #[test]
    pub fn test_cube_activation() {
        let function = ActivationFn::Cube;

        assert_approx_eq!(function.eval(0.0), 0.0);
        assert_approx_eq!(function.eval(2.0), 8.0);
        assert_approx_eq!(function.eval(-2.0), -8.0);
    }
}
