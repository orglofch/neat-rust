use std::f32;

#[derive(Clone, Copy)]
pub enum AggregationFn {
    Sum,
    Product,
    Max,
    Min,
    AbsMax,
    // TODO(orglofch): Median
    Mean,
}

impl AggregationFn {
    pub(crate) fn aggregate(&self, values: Vec<f32>) -> f32 {
        match *self {
            AggregationFn::Sum => sum_aggregation(values),
            AggregationFn::Product => product_aggregation(values),
            AggregationFn::Max => max_aggregation(values),
            AggregationFn::Min => min_aggregation(values),
            AggregationFn::AbsMax => abs_max_aggregation(values),
            AggregationFn::Mean => mean_aggregation(values),
        }
    }
}

#[inline]
fn sum_aggregation(values: Vec<f32>) -> f32 {
    values.iter().sum()
}

#[inline]
fn product_aggregation(values: Vec<f32>) -> f32 {
    values.iter().product()
}

#[inline]
fn max_aggregation(values: Vec<f32>) -> f32 {
    values.into_iter().fold(f32::MIN, f32::max)
}

#[inline]
fn min_aggregation(values: Vec<f32>) -> f32 {
    values.into_iter().fold(f32::MAX, f32::min)
}

#[inline]
fn abs_max_aggregation(values: Vec<f32>) -> f32 {
    values.into_iter().map(f32::abs).fold(f32::MIN, f32::max)
}

#[inline]
fn mean_aggregation(values: Vec<f32>) -> f32 {
    let count = values.len() as f32;
    if count == 0.0 {
        0.0
    } else {
        sum_aggregation(values) / count
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_sum_aggregation() {
        let function = AggregationFn::Sum;

        assert_approx_eq!(function.aggregate(vec![]), 0.0);
        assert_approx_eq!(function.aggregate(vec![1.0, 2.0, 3.0, 4.0]), 10.0);
        assert_approx_eq!(function.aggregate(vec![-1.0, 2.0, -3.0, 4.0]), 2.0);
    }

    #[test]
    pub fn test_prod_aggregation() {
        let function = AggregationFn::Product;

        assert_approx_eq!(function.aggregate(vec![]), 1.0);
        assert_approx_eq!(function.aggregate(vec![-1.0, 2.0, 2.0]), -4.0);
        assert_approx_eq!(function.aggregate(vec![1.0, 2.0, 0.0]), 0.0);
    }

    #[test]
    pub fn test_max_aggregation() {
        let function = AggregationFn::Max;

        assert_approx_eq!(function.aggregate(vec![]), f32::MIN);
        assert_approx_eq!(function.aggregate(vec![3.0, 1.0, 2.0]), 3.0);
        assert_approx_eq!(function.aggregate(vec![-1.0, -2.0, -10.0]), -1.0);
    }

    #[test]
    pub fn test_min_aggregation() {
        let function = AggregationFn::Min;

        assert_approx_eq!(function.aggregate(vec![]), f32::MAX);
        assert_approx_eq!(function.aggregate(vec![3.0, 1.0, 2.0]), 1.0);
        assert_approx_eq!(function.aggregate(vec![-1.0, -2.0, -10.0]), -10.0);
    }

    #[test]
    pub fn test_max_abs_aggregation() {
        let function = AggregationFn::AbsMax;

        assert_approx_eq!(function.aggregate(vec![]), f32::MIN);
        assert_approx_eq!(function.aggregate(vec![3.0, 1.0, 2.0]), 3.0);
        assert_approx_eq!(function.aggregate(vec![-1.0, -2.0, -10.0]), 10.0);
    }

    #[test]
    pub fn test_mean_aggregation() {
        let function = AggregationFn::Mean;

        assert_approx_eq!(function.aggregate(vec![]), 0.0);
        assert_approx_eq!(function.aggregate(vec![2.0, 1.0, 3.0, 4.0]), 2.5);
        assert_approx_eq!(function.aggregate(vec![-1.0, -2.0, 1.0, 2.0]), 0.0);
    }
}
