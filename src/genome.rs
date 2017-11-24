extern crate rand;

use self::rand::Rng;

use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;

use activation::ActivationFn;
use innovation::InnovationArchive;
use speciation::SpeciationConfig;

// TODO(orglofch): Consider separating initial and mutation configs.
/// The configuration controlling genome creation and mutation.
///
/// # Note
///
/// Tthe sum of mutation probability need not necessary add up to 1.
/// If `sum < 1` there wil be a chance of no mutation being taken.
/// If `sum > 1` the probability for a single mutation will be relative to the total.
pub struct GenomeConfig {
    innovation_archive: InnovationArchive,

    // TODO(orglofch): Allow inputs and outputs to be named, making the addition or removal
    // of input or outputs in checkpointing more explicit.
    /// The number of input nodes, excluding bias.
    num_inputs: u32,

    /// The nunber of output nodes.
    num_outputs: u32,

    // TODO(orglofch): Allow for configuration that have multiple mutations
    // for every mutation iteration.
    /// Whether the input should start connected to the outputs.
    ///
    /// # Note
    ///
    /// In some circumstances having the outputs initially unconnected can
    /// improve the ability to prune unnecessary dependencies.
    start_connected: bool,

    /// The probability of mutating the addition of a connection.
    mutate_add_con_prob: f32,

    /// The probability of mutating the removal of a connection.
    mutate_remove_con_prob: f32,

    /// The probability of mutating the addition of a node.
    mutate_add_node_prob: f32,

    /// The probability of mutating the removal of a node.
    mutate_remove_node_prob: f32,
}

impl GenomeConfig {
    pub fn new(num_inputs: u32, num_outputs: u32) -> GenomeConfig {
        GenomeConfig {
            innovation_archive: InnovationArchive::new(),
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            start_connected: false,
            mutate_add_con_prob: 0.5,
            mutate_remove_con_prob: 0.0,
            mutate_add_node_prob: 0.5,
            mutate_remove_node_prob: 0.0,
        }
    }

    pub fn set_num_inputs(&mut self, inputs: u32) -> &mut GenomeConfig {
        self.num_inputs = inputs;
        self
    }

    pub fn set_num_outputs(&mut self, outputs: u32) -> &mut GenomeConfig {
        self.num_outputs = outputs;
        self
    }

    pub fn set_start_connected(&mut self, enabled: bool) -> &mut GenomeConfig {
        self.start_connected = enabled;
        self
    }

    pub fn set_mutate_add_connection_probability(&mut self, probability: f32) -> &mut GenomeConfig {
        self.mutate_add_con_prob = probability;
        self
    }

    pub fn set_mutate_remove_connection_probability(
        &mut self,
        probability: f32,
    ) -> &mut GenomeConfig {
        self.mutate_remove_con_prob = probability;
        self
    }

    pub fn set_mutate_add_node_probability(&mut self, probability: f32) -> &mut GenomeConfig {
        self.mutate_add_node_prob = probability;
        self
    }

    pub fn set_mutate_remove_node_probability(&mut self, probability: f32) -> &mut GenomeConfig {
        self.mutate_remove_node_prob = probability;
        self
    }
}

#[derive(Clone)]
pub enum AggregationFn {
    Sum,
}

impl AggregationFn {
    fn aggregate(self, vals: &Vec<f32>) -> f32 {
        match self {
            AggregationFn::Sum => vals.iter().sum(),
        }
    }
}


#[derive(Clone)]
pub struct NodeGene {
    // TODO(orglofch): Consider combining activation and aggregation.
    /// The function used in aggregating across synapses prior to calculating activation.
    aggregation_fn: AggregationFn,

    /// The function used inaga determining neuron activation.
    activation_fn: ActivationFn,
}

impl NodeGene {
    fn new(aggregation_fn: AggregationFn, activation_fn: ActivationFn) -> NodeGene {
        NodeGene {
            aggregation_fn: aggregation_fn,
            activation_fn: activation_fn,
        }
    }

    // TODO(orglofch): Include a distance function for this as well.
}

#[derive(Clone)]
pub struct ConnectionGene {
    /// The id of the input `NodeGene`.
    in_id: u32,

    /// The id of the output `NodeGene`.
    out_id: u32,

    /// Whether the connection is enabled.
    ///
    /// Connections can be disabled when being split by the creation of new nodes or through
    /// mutation but are persisted in the genome as a means of maintaining ancestry.
    enabled: bool,

    /// The synaptic excitement applied to the input to produce the output.
    weight: f32,

    /// The number of times this connection has been split.
    splits: u32,
}

impl ConnectionGene {
    fn new(in_id: u32, out_id: u32, weight: f32) -> ConnectionGene {
        ConnectionGene {
            in_id: in_id,
            out_id: out_id,
            enabled: true,
            weight: weight,
            splits: 0,
        }
    }

    /// Calculates the compatibility distance between this `ConnectionGene` and another.
    fn distance(&self, other: &ConnectionGene, speciation_config: &SpeciationConfig) -> f32 {
        // TODO(orglofch): Consider concept of disabled state.
        return (self.weight - other.weight).abs() *
            speciation_config.compatibility_weight_coefficient;
    }
}

pub struct Genome {
    id: u32,

    /// `NodeGenes` by their id.
    nodes_by_id: HashMap<u32, NodeGene>,

    /// `ConnectionGene` by their id.
    connections_by_id: HashMap<u32, ConnectionGene>,
}

impl Genome {
    pub(crate) fn new(id: u32, config: &mut GenomeConfig) -> Genome {
        let mut nodes_by_id: HashMap<u32, NodeGene> =
            HashMap::with_capacity(config.num_inputs as usize + config.num_outputs as usize + 1);

        let num_connections = if config.start_connected {
            config.num_inputs as usize * config.num_inputs as usize
        } else {
            0
        };

        let mut connections_by_id: HashMap<u32, ConnectionGene> =
            HashMap::with_capacity(num_connections);

        // Create bias node which always has id 0 in all configuration.
        // TODO(orglofch): Make the bias id a constant which is obeyed by the innovation archive.
        let bias_id = config.innovation_archive.record_spontaneous_input_node(0);
        nodes_by_id.insert(
            bias_id,
            NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid),
        );

        // Create input sensor nodes.
        for input_id in 0..config.num_inputs {
            let id = config.innovation_archive.record_spontaneous_input_node(
                input_id + 1,
            );
            nodes_by_id.insert(id, NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid));
        }

        // Create output nodes.
        for output_id in 0..config.num_outputs {
            let id = config.innovation_archive.record_spontaneous_output_node(
                output_id,
            );
            nodes_by_id.insert(id, NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid));
        }

        // If the inputs and outputs start connected then create the initial ConnectionGenes
        // which connect each input to each output.
        if config.start_connected {
            for input_id in 0..config.num_inputs + 1 {
                for output_id in 0..config.num_outputs {
                    let in_id = config.innovation_archive.record_spontaneous_input_node(
                        input_id,
                    );
                    let out_id = config.innovation_archive.record_spontaneous_output_node(
                        output_id,
                    );

                    let id = config.innovation_archive.record_connection_innovation(
                        in_id,
                        out_id,
                    );
                    connections_by_id.insert(id, ConnectionGene::new(in_id, out_id, 1.0));
                }
            }
        }

        Genome {
            id: id,
            nodes_by_id: nodes_by_id,
            connections_by_id: connections_by_id,
        }
    }

    /// Mutates the `Genome`.
    ///
    /// # Arguments
    ///
    /// * `genome_config` - The configuration governing the mutation of the `Genome`.
    pub(crate) fn mutate<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let sum_prob = genome_config.mutate_add_con_prob + genome_config.mutate_remove_con_prob +
            genome_config.mutate_add_node_prob +
            genome_config.mutate_remove_node_prob;

        // If the sum of the probabilities is greater than 1, normalize the probability range
        // so the sum consistutes a 100% probability. Otherwise, allow the total probability
        // to be less than 1 to allow for a probability of no mutations.
        let prob = rng.gen_range::<f32>(0.0, sum_prob.max(1.0));

        if prob < genome_config.mutate_add_con_prob {
            self.mutate_add_connection(genome_config, rng);
        } else if prob < genome_config.mutate_remove_con_prob {
            // TODO(orglofch): Implement.
        } else if prob < genome_config.mutate_add_node_prob {
            self.mutate_add_node(genome_config, rng);
        } else if prob < genome_config.mutate_remove_node_prob {
            // TODO(orglofch): Implement.
        }

        // TODO(orglofch): Mutate weight.
    }

    /// Calculates the compatibility distance between this `Genome` and another.
    ///
    /// # Arguments
    ///
    /// `other` - The other `Genome` to compare against.
    ///
    /// `speciation_config` - The configuration governing the equation used in calculating the distance.
    pub(crate) fn distance(&self, other: &Genome, speciation_config: &SpeciationConfig) -> f32 {
        // TODO(orglofch): Play around with other gene distance functions this currently treats excess
        // and disjoint genes as being equivalent.

        // NodeGene distance.
        let disjoint_nodes = {
            let mut disjoint_genes = 0_u32;

            for id in self.nodes_by_id.keys() {
                if !other.nodes_by_id.contains_key(&id) {
                    disjoint_genes += 1;
                }
            }

            for id in other.nodes_by_id.keys() {
                if !self.nodes_by_id.contains_key(&id) {
                    disjoint_genes += 1;
                }
            }

            disjoint_genes
        };

        // ConnectionGene distance.
        let (disjoint_connections, common_distance) = {
            let mut disjoint_genes = 0_u32;
            let mut common_distance = 0.0;

            for (id, con) in self.connections_by_id.iter() {
                match other.connections_by_id.get(&id) {
                    Some(ref other_con) => {
                        common_distance += con.distance(other_con, speciation_config);
                    }
                    None => {
                        disjoint_genes += 1;
                    }
                }
            }

            for id in other.connections_by_id.keys() {
                if !self.connections_by_id.contains_key(&id) {
                    disjoint_genes += 1;
                }
            }

            (disjoint_genes, common_distance)
        };

        let normalization_constant = {
            let self_gene_count = self.nodes_by_id.len() + self.connections_by_id.len();
            let other_gene_count = other.nodes_by_id.len() + other.connections_by_id.len();

            // If the genes are sufficient small then don't normalize.
            if self_gene_count < 20 && other_gene_count < 20 {
                1.0
            } else {
                // Use the large of the two gene counts.
                cmp::max(self_gene_count, other_gene_count) as f32
            }
        };

        let disjoint_distance = (disjoint_nodes + disjoint_connections) as f32 *
            speciation_config.compatibility_disjoint_coefficient /
            normalization_constant;

        disjoint_distance + common_distance
    }

    /// Mutates the `Genome` by adding a new `ConnectionGene` structural mutation.
    fn mutate_add_connection<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let active_connections: HashSet<(u32, u32)> = self.connections_by_id
            .values()
            .filter(|con| con.enabled)
            .map(|con| (con.in_id, con.out_id))
            .collect();

        let mut possible_connections: Vec<(u32, u32)> = Vec::new();
        for i in 0..self.nodes_by_id.len() as u32 {
            for j in 0..self.nodes_by_id.len() as u32 {
                if i == j || active_connections.contains(&(i, j)) {
                    continue;
                }
                possible_connections.push((i, j));
            }
        }

        if possible_connections.is_empty() {
            return;
        }

        // TODO(orglofch): Ensure you can't pair to the bias (as output) probably.

        // TODO(orglofch): This doesn't work right now since it can pick two nodes which are connected.
        // We should try finding all pairs, then select one.

        let connection = rng.choose(&mut possible_connections).unwrap();

        // Check if the connection already exists but is disabled.
        {
            let maybe_existing_connection = self.connections_by_id.values_mut().find(|ref con| {
                con.in_id == connection.0 && con.out_id == connection.1
            });
            if maybe_existing_connection.is_some() {
                maybe_existing_connection.unwrap().enabled = true;
                return;
            }
        }

        // TODO(orglofch): Random weight?
        let connection_id = genome_config
            .innovation_archive
            .record_connection_innovation(connection.0, connection.1);
        self.connections_by_id.insert(
            connection_id,
            ConnectionGene::new(connection.0, connection.1, 1.0),
        );
    }

    /// Mutate the `Genome` by adding a new `NodeGene` structural mutation.
    fn mutate_add_node<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let maybe_connection = self.split_random_connection(rng);
        if maybe_connection.is_none() {
            return;
        }

        let con = maybe_connection.unwrap();

        // Create a new hidden node.
        let new_id = genome_config
            .innovation_archive
            .record_hidden_node_innovation(con.in_id, con.out_id, con.splits);
        self.nodes_by_id.insert(
            new_id,
            NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid),
        );

        // Create two new connections.
        // The first connection is initialized with a weight of 1.
        // The second connection is initilized with the original connections weight.
        // This preserves the original phenotype.
        let new_connection_id_1 = genome_config
            .innovation_archive
            .record_connection_innovation(con.in_id, new_id);
        let new_connection_id_2 = genome_config
            .innovation_archive
            .record_connection_innovation(new_id, con.out_id);

        self.connections_by_id.insert(
            new_connection_id_1,
            ConnectionGene::new(con.in_id, new_id, 1.0),
        );
        self.connections_by_id.insert(
            new_connection_id_2,
            ConnectionGene::new(new_id, con.out_id, con.weight),
        );
    }

    /// Mutate the `Genome` by modifying the weight on an existing `NodeGene`.
    fn mutate_weight<R: Rng>(&mut self, rng: &mut R) {
        // Select a random active connection.
        let mut active_connections: Vec<&mut ConnectionGene> = self.connections_by_id
            .values_mut()
            .filter(|con| con.enabled)
            .collect();
        if active_connections.is_empty() {
            return;
        }

        let mut connection = rng.choose_mut(&mut active_connections).unwrap();

        // TODO(orglofch): Make the distribution variable and move the current weight
        // rather than setting a new weight, possibly based on a covariance.
        connection.weight = rng.gen_range::<f32>(0.0, 1.0);
    }

    /// Splits a random enabled connection and returns a copy of that gene.
    fn split_random_connection<R: Rng>(&mut self, rng: &mut R) -> Option<ConnectionGene> {
        let mut enabled_connections: Vec<&mut ConnectionGene> = self.connections_by_id
            .values_mut()
            .filter(|con| con.enabled)
            .collect();
        if enabled_connections.is_empty() {
            return None;
        }

        let mut connection = rng.choose_mut(&mut enabled_connections).unwrap();

        // Disable the original connection.
        connection.enabled = false;

        // Increase the number of splits.
        connection.splits += 1;

        return Some(connection.clone());
    }
}

pub type Population = Vec<Genome>;

// TODO(orglofch): More comprehensive equality tests.
// TODO(orglofch): Figure out how to force seed the rng without implementing the whole trait.
// TODO(orglofch): Adds tests for mutate weights.
#[cfg(test)]
mod test {

    #[test]
    fn test_distance_connection() {
        let mut genome_config = super::GenomeConfig::new(0, 1);
        genome_config.set_start_connected(true);

        let mut speciation_config = super::SpeciationConfig::new();
        speciation_config.set_compatibility_weight_coefficient(1.0);

        let mut genome_1 = super::Genome::new(0, &mut genome_config);
        let mut genome_2 = super::Genome::new(0, &mut genome_config);

        genome_1.connections_by_id.get_mut(&0).unwrap().weight = 1.0;
        genome_2.connections_by_id.get_mut(&0).unwrap().weight = 1.0;

        // Same weights.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);

        genome_1.connections_by_id.get_mut(&0).unwrap().weight = 0.5;

        // Weights differ.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.5);

        speciation_config.set_compatibility_weight_coefficient(0.5);

        // Weights differ and coefficient is fractional.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.25);

        genome_2.connections_by_id.get_mut(&0).unwrap().weight = 0.5;

        // Weights are the same but coefficient is fractional.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);
    }

    #[test]
    fn test_distance_disjoint() {
        let mut genome_config = super::GenomeConfig::new(0, 1);

        let mut speciation_config = super::SpeciationConfig::new();
        speciation_config.set_compatibility_disjoint_coefficient(1.0);

        let mut genome_1 = super::Genome::new(0, &mut genome_config);
        let mut genome_2 = super::Genome::new(1, &mut genome_config);

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);

        // Nodes are disjoint.
        genome_1.nodes_by_id.insert(
            2,
            super::NodeGene::new(
                super::AggregationFn::Sum,
                super::ActivationFn::Sigmoid,
            ),
        );

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 1.0);

        // Nodes and connections are disjoint.
        genome_2.connections_by_id.insert(
            2,
            super::ConnectionGene::new(
                1,
                2,
                1.0,
            ),
        );

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 2.0);

        // Nodes and connections are disjoint and coefficient is fractional.
        speciation_config.set_compatibility_disjoint_coefficient(0.5);

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 1.0);

        // Connections are disjoint.
        genome_2.nodes_by_id.insert(
            2,
            super::NodeGene::new(
                super::AggregationFn::Sum,
                super::ActivationFn::Sigmoid,
            ),
        );

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.5);
    }

    #[test]
    fn test_new_genome_unconnected() {
        let mut config = super::GenomeConfig::new(1, 1);

        let genome = super::Genome::new(1, &mut config);

        assert_eq!(genome.id, 1);
        assert_eq!(genome.connections_by_id.len(), 0);
        assert_eq!(genome.nodes_by_id.len(), 3);

        config.set_num_inputs(2);

        let genome = super::Genome::new(2, &mut config);

        assert_eq!(genome.id, 2);
        assert_eq!(genome.connections_by_id.len(), 0);
        assert_eq!(genome.nodes_by_id.len(), 4);

        config.set_num_outputs(2);

        let genome = super::Genome::new(3, &mut config);

        assert_eq!(genome.id, 3);
        assert_eq!(genome.connections_by_id.len(), 0);
        assert_eq!(genome.nodes_by_id.len(), 5);
    }

    #[test]
    fn test_new_genome_start_connected() {
        let mut config = super::GenomeConfig::new(1, 1);
        config.set_start_connected(true);

        let genome = super::Genome::new(1, &mut config);

        assert_eq!(genome.id, 1);
        assert_eq!(genome.connections_by_id.len(), 2);
        assert_eq!(genome.nodes_by_id.len(), 3);

        config.set_num_inputs(2);

        let genome = super::Genome::new(2, &mut config);

        assert_eq!(genome.id, 2);
        assert_eq!(genome.connections_by_id.len(), 3);
        assert_eq!(genome.nodes_by_id.len(), 4);

        config.set_num_outputs(2);

        let genome = super::Genome::new(3, &mut config);

        assert_eq!(genome.id, 3);
        assert_eq!(genome.connections_by_id.len(), 6);
        assert_eq!(genome.nodes_by_id.len(), 5);
    }

    #[test]
    fn test_mutate_add_connection_new() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(false);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &mut config);

        genome.mutate_add_connection(&mut config, &mut rng);

        assert_eq!(genome.connections_by_id.len(), 1);
        assert_eq!(genome.nodes_by_id.len(), 2);
    }

    #[test]
    fn test_mutate_add_connection_no_free_connection() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(false);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &mut config);

        // Fill the only possible connection.
        let connection = super::ConnectionGene::new(1, 0, 1.0);
        genome.connections_by_id.insert(2, connection);

        genome.mutate_add_connection(&mut config, &mut rng);

        assert_eq!(genome.connections_by_id.len(), 2);
        assert_eq!(genome.nodes_by_id.len(), 2);
    }

    // TODO(orglofch): fn test_mutate_add_connection_reenable() {}

    #[test]
    fn test_mutate_add_node() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(true);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &mut config);

        genome.mutate_add_node(&mut config, &mut rng);

        assert_eq!(genome.connections_by_id.len(), 3);
        assert_eq!(genome.nodes_by_id.len(), 3);
    }

    #[test]
    fn test_mutate_add_node_no_free_connection() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(false);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &mut config);

        genome.mutate_add_node(&mut config, &mut rng);

        assert_eq!(genome.connections_by_id.len(), 0);
        assert_eq!(genome.nodes_by_id.len(), 2);
    }
}
