use activation::ActivationFn;
use aggregation::AggregationFn;
use speciation::SpeciationConfig;

#[derive(Clone, Debug)]
pub struct NodeGene {
    /// The function used in aggregating across synapses prior to calculating activation.
    pub(crate) aggregation_fn: AggregationFn,

    /// The function used inaga determining neuron activation.
    pub(crate) activation_fn: ActivationFn,

    /// The bias applied to the node.
    ///
    /// We store the bias in the `NodeGene` instead of via a `ConnectionGene` since
    /// we don't want bias connection to be splittable and bias is generally fully connected
    /// or disabled (via `bias = 0.0`).
    pub(crate) bias: f32,
}

impl NodeGene {
    pub(crate) fn new(aggregation_fn: AggregationFn, activation_fn: ActivationFn) -> NodeGene {
        NodeGene {
            aggregation_fn: aggregation_fn,
            activation_fn: activation_fn,
            bias: 0.0,
        }
    }

    /// Calculates the compatibiltiy distance between this `NodeGene` and another.
    pub(crate) fn distance(&self, other: &NodeGene, speciation_config: &SpeciationConfig) -> f32 {
        return (self.bias - other.bias).abs() * speciation_config.compatibility_weight_coefficient;
    }

    /// Calculates the activation of the `NodeGene` given it's inputs.
    pub(crate) fn activate(&self, values: &Vec<f32>) -> f32 {
        let mut values = values.clone();

        // Include bias.
        values.push(self.bias);

        // Perform aggregation on the inputs.
        let aggregation = self.aggregation_fn.aggregate(values);

        self.activation_fn.eval(aggregation)
    }

    /// Create a new `NodeGene` via crossover between this `NodeGene` and another.
    pub(crate) fn crossover(&self, other: &NodeGene) -> NodeGene {
        // TODO(orglofch): Better crossover.
        NodeGene {
            aggregation_fn: self.aggregation_fn,
            activation_fn: self.activation_fn,
            bias: (self.bias + other.bias) / 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConnectionGene {
    /// Whether the connection is enabled.
    ///
    /// Connections can be disabled when being split by the creation of new nodes or through
    /// mutation but are persisted in the genome as a means of maintaining ancestry.
    pub(crate) enabled: bool,

    /// The synaptic excitement applied to the input to produce the output.
    pub(crate) weight: f32,

    /// The number of times this connection has been split.
    pub(crate) splits: u32,
}

impl ConnectionGene {
    pub(crate) fn new(weight: f32) -> ConnectionGene {
        ConnectionGene {
            enabled: true,
            weight: weight,
            splits: 0,
        }
    }

    /// Calculates the compatibility distance between this `ConnectionGene` and another.
    pub(crate) fn distance(&self, other: &ConnectionGene, speciation_config: &SpeciationConfig) -> f32 {
        // TODO(orglofch): Consider concept of disabled state.
        return (self.weight - other.weight).abs() * speciation_config.compatibility_weight_coefficient;
    }

    /// Calculates the activation of the `ConnectionGene` given it's input.
    pub(crate) fn activate(&self, value: f32) -> f32 {
        value * self.weight
    }

    /// Creae a new `ConnectionGene` via crossover between this `ConnectionGene` and another.
    pub(crate) fn crossover(&self, other: &ConnectionGene) -> ConnectionGene {
        debug_assert_eq!(self.splits, other.splits, "Only the same gene should be crossed");

        // TODO(orglofch): Better crossover.
        ConnectionGene {
            enabled: self.enabled,
            weight: (self.weight + other.weight) / 2.0,
            splits: self.splits,
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_node_distance() {
        let mut spec_conf = SpeciationConfig::new();
        spec_conf.set_compatibility_weight_coefficient(1.0);

        let mut node_1 = NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid);
        let mut node_2 = NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid);

        node_1.bias = 1.0;
        node_2.bias = 1.0;

        // Same bias.
        assert_eq!(
            node_1.distance(&node_2, &spec_conf),
            node_2.distance(&node_1, &spec_conf)
        );
        assert_eq!(node_1.distance(&node_2, &spec_conf), 0.0);

        node_1.bias = 0.5;

        // Bias differs.
        assert_eq!(
            node_1.distance(&node_2, &spec_conf),
            node_2.distance(&node_1, &spec_conf)
        );
        assert_eq!(node_1.distance(&node_2, &spec_conf), 0.5);

        spec_conf.set_compatibility_weight_coefficient(0.5);

        // Bias differ and coefficient is fractional.
        assert_eq!(
            node_1.distance(&node_2, &spec_conf),
            node_2.distance(&node_1, &spec_conf)
        );
        assert_eq!(node_1.distance(&node_2, &spec_conf), 0.25);

        node_2.bias = 0.5;

        // Bias is the same but coefficient is fractional.
        assert_eq!(
            node_1.distance(&node_2, &spec_conf),
            node_2.distance(&node_1, &spec_conf)
        );
        assert_eq!(node_1.distance(&node_2, &spec_conf), 0.0);
    }

    #[test]
    fn test_node_crossover() {
        // TODO(orglofch): Implement.
    }

    #[test]
    fn test_connection_distance() {
        let mut spec_conf = SpeciationConfig::new();
        spec_conf.set_compatibility_weight_coefficient(1.0);

        let mut con_1 = ConnectionGene::new(1.0);
        let mut con_2 = ConnectionGene::new(1.0);

        // Same weights.
        assert_eq!(
            con_1.distance(&con_2, &spec_conf),
            con_2.distance(&con_1, &spec_conf)
        );
        assert_eq!(con_1.distance(&con_2, &spec_conf), 0.0);

        con_1.weight = 0.5;

        // Weights differ.
        assert_eq!(
            con_1.distance(&con_2, &spec_conf),
            con_2.distance(&con_1, &spec_conf)
        );
        assert_eq!(con_1.distance(&con_2, &spec_conf), 0.5);

        spec_conf.set_compatibility_weight_coefficient(0.5);

        // Weights differ and coefficient is fractional.
        assert_eq!(
            con_1.distance(&con_2, &spec_conf),
            con_2.distance(&con_1, &spec_conf)
        );
        assert_eq!(con_1.distance(&con_2, &spec_conf), 0.25);

        con_2.weight = 0.5;

        // Weights are the same but coefficient is fractional.
        assert_eq!(
            con_1.distance(&con_2, &spec_conf),
            con_2.distance(&con_1, &spec_conf)
        );
        assert_eq!(con_1.distance(&con_2, &spec_conf), 0.0);
    }

    #[test]
    fn test_connection_crossover() {
        // TODO(orglofch): Implement.
    }
}
