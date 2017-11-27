use std::f32;

#[derive(Clone)]
pub enum AggregationFn {
    Sum,
    Product,
    Max,
    Min,
    MaxAbs,
    Median,
    Mean,
}

impl AggregationFn {
    pub(crate) fn aggregate(&self, values: Vec<f32>) -> f32 {
        match *self {
            AggregationFn::Sum => sum_aggregation(values),
            AggregationFn::Product => product_aggregation(values),
            AggregationFn::Max => max_aggregation(values),
            AggregationFn::Min => min_aggregation(values),
            AggregationFn::MaxAbs => max_abs_aggregation(values),
            AggregationFn::Median => median_aggregation(values),
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
fn max_abs_aggregation(values: Vec<f32>) -> f32 {
    values.into_iter().map(f32::abs).fold(f32::MAX, f32::min)
}

#[inline]
fn median_aggregation(values: Vec<f32>) -> f32 {
    panic!("TODO(orglofch): Implement");
}

#[inline]
fn mean_aggregation(values: Vec<f32>) -> f32 {
    let count = values.len() as f32;
    sum_aggregation(values) / count
}
