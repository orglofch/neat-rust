#[derive(Clone)]
pub enum AggregationFn {
    Sum,
}

impl AggregationFn {
    pub(crate) fn aggregate(&self, vals: Vec<f32>) -> f32 {
        match self {
            &AggregationFn::Sum => vals.iter().sum(),
        }
    }
}
