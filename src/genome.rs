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
/// If `sum < 1` there will be a chance of no mutation being taken.
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
    id: u32,

    // TODO(orglofch): Consider combining activation and aggregation.
    /// The function used in aggregating across synapses prior to calculating activation.
    ///
    /// Only set when `kind = NodeType::Hidden || NodeType::Output`.
    aggregation_fn: Option<AggregationFn>,

    /// The function used inaga determining neuron activation.
    ///
    /// Only set when `kind = NodeType::Hidden || NodeType::Output`.
    activation_fn: Option<ActivationFn>,
}

impl NodeGene {
    // TODO(orglofch): Include a distance function for this as well.
}

#[derive(Clone)]
pub struct ConnectionGene {
    innovation_id: u32,

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
    fn new(id: u32, in_id: u32, out_id: u32, weight: f32) -> ConnectionGene {
        ConnectionGene {
            innovation_id: id,
            in_id: in_id,
            out_id: out_id,
            enabled: true,
            weight: weight,
            splits: 0,
        }
    }

    /// Calculates the compatibility distance between this `ConnectionGene` and another.
    fn distance(&self, other: &ConnectionGene) -> f32 {
        // TODO(orglofch): Consider concept of disabled state.
        return (self.weight - other.weight).abs();
    }
}

pub struct Genome {
    id: u32,

    nodes: Vec<NodeGene>,

    /// `ConnectionGene` by their
    // TODO(orglofch): Make this a map.
    connections: Vec<ConnectionGene>,
}

impl Genome {
    pub(crate) fn new(id: u32, config: &GenomeConfig) -> Genome {
        let mut nodes: Vec<NodeGene> =
            Vec::with_capacity(config.num_inputs as usize + config.num_outputs as usize + 1);

        let num_connections = if config.start_connected {
            config.num_inputs as usize * config.num_inputs as usize
        } else {
            0
        };

        let mut connections: Vec<ConnectionGene> = Vec::with_capacity(num_connections);

        // Create bias node which always has id 0 in all configuration.
        nodes.push(NodeGene {
            id: 0,
            activation_fn: None,
            aggregation_fn: None,
        });

        // End positions of NodeGene id ranges.
        let input_end = config.num_inputs + 1;
        let output_end = input_end + config.num_outputs;

        // Create input sensor nodes.
        for id in 1..input_end {
            nodes.push(NodeGene {
                id: id + 1,
                activation_fn: None,
                aggregation_fn: None,
            })
        }

        // Create output nodes.
        for id in input_end..output_end {
            nodes.push(NodeGene {
                id: id + config.num_inputs + 1,
                aggregation_fn: Some(AggregationFn::Sum),
                activation_fn: Some(ActivationFn::Sigmoid),
            })
        }

        // If the inputs and outputs start connected then create the initial ConnectionGenes
        // which connect each input to each output.
        if config.start_connected {
            for in_id in 0..input_end {
                for out_id in input_end..output_end {
                    connections.push(ConnectionGene::new(
                        in_id + out_id * config.num_inputs,
                        in_id,
                        out_id,
                        1.0,
                    ));
                }
            }
        }

        Genome {
            id: id,
            nodes: nodes,
            connections: connections,
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
    }

    /// Mutates the `Genome` by adding a new `ConnectionGene` structural mutation.
    fn mutate_add_connection<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let active_connections: HashSet<(u32, u32)> = self.connections
            .iter()
            .filter(|con| con.enabled)
            .map(|con| (con.in_id, con.out_id))
            .collect();

        let mut possible_connections: Vec<(u32, u32)> = Vec::new();
        for i in 0..self.nodes.len() as u32 {
            for j in 0..self.nodes.len() as u32 {
                if i == j || active_connections.contains(&(i, j)) {
                    continue;
                }
                possible_connections.push((i, j));
            }
        }

        if possible_connections.is_empty() {
            return;
        }

        // TODO(orglofch): This doesn't work right now since it can pick two nodes which are connected.
        // We should try finding all pairs, then select one.

        let connection = rng.choose(&mut possible_connections).unwrap();

        // Check if the connection already exists but is disabled.
        {
            let maybe_existing_connection = self.connections.iter_mut().find(|ref con| {
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
        self.connections.push(ConnectionGene::new(
            connection_id,
            connection.0,
            connection.1,
            1.0,
        ));
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
        self.nodes.push(NodeGene {
            id: new_id,
            aggregation_fn: Some(AggregationFn::Sum),
            activation_fn: Some(ActivationFn::Sigmoid),
        });

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

        self.connections.push(ConnectionGene::new(
            new_connection_id_1,
            con.in_id,
            new_id,
            1.0,
        ));
        self.connections.push(ConnectionGene::new(
            new_connection_id_2,
            new_id,
            con.out_id,
            con.weight,
        ));
    }

    /// Mutate the `Genome` by modifying the weight on an existing `NodeGene`.
    fn mutate_weight<R: Rng>(&mut self, rng: &mut R) {
        // Select a random active connection.
        let mut active_connections: Vec<&mut ConnectionGene> = self.connections
            .iter_mut()
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

    /// Calculates the compatibility distance between this `Genome` and another.
    ///
    /// # Arguments
    ///
    /// `other` - The other `Genome` to compare against.
    ///
    /// `speciation_config` - The configuration governing the equation used in calculating the distance.
    fn distance(&self, other: Genome, speciation_config: &SpeciationConfig) -> f32 {
        // TODO(orglofch): Play around with other gene distance functions this currently treats excess
        // and disjoint genes as being equivalent.

        // NodeGene distance.
        let node_gene_distance = {
            let mut disjoint_nodes = 0;
            for node in other.nodes.iter() {}
        };

        0.0
    }

    /// Splits a random enabled connection and returns a copy of that gene.
    fn split_random_connection<R: Rng>(&mut self, rng: &mut R) -> Option<ConnectionGene> {
        let mut enabled_connections: Vec<&mut ConnectionGene> = self.connections
            .iter_mut()
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
    fn test_new_genome_unconnected() {
        let mut config = super::GenomeConfig::new(1, 1);

        let genome = super::Genome::new(1, &config);

        assert_eq!(genome.id, 1);
        assert_eq!(genome.connections.len(), 0);
        assert_eq!(genome.nodes.len(), 3);

        config.set_num_inputs(2);

        let genome = super::Genome::new(2, &config);

        assert_eq!(genome.id, 2);
        assert_eq!(genome.connections.len(), 0);
        assert_eq!(genome.nodes.len(), 4);

        config.set_num_outputs(2);

        let genome = super::Genome::new(3, &config);

        assert_eq!(genome.id, 3);
        assert_eq!(genome.connections.len(), 0);
        assert_eq!(genome.nodes.len(), 5);
    }

    #[test]
    fn test_new_genome_start_connected() {
        let mut config = super::GenomeConfig::new(1, 1);
        config.set_start_connected(true);

        let genome = super::Genome::new(1, &config);

        assert_eq!(genome.id, 1);
        assert_eq!(genome.connections.len(), 2);
        assert_eq!(genome.nodes.len(), 3);

        config.set_num_inputs(2);

        let genome = super::Genome::new(2, &config);

        assert_eq!(genome.id, 2);
        assert_eq!(genome.connections.len(), 3);
        assert_eq!(genome.nodes.len(), 4);

        config.set_num_outputs(2);

        let genome = super::Genome::new(3, &config);

        assert_eq!(genome.id, 3);
        assert_eq!(genome.connections.len(), 6);
        assert_eq!(genome.nodes.len(), 5);
    }

    #[test]
    fn test_mutate_add_connection_new() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(false);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &config);

        genome.mutate_add_connection(&mut config, &mut rng);

        assert_eq!(genome.connections.len(), 1);
        assert_eq!(genome.nodes.len(), 2);
    }

    #[test]
    fn test_mutate_add_connection_no_free_connection() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(false);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &config);

        // Fill the only possible connection.
        let connection = super::ConnectionGene::new(2, 1, 0, 1.0);
        genome.connections.push(connection);

        genome.mutate_add_connection(&mut config, &mut rng);

        assert_eq!(genome.connections.len(), 2);
        assert_eq!(genome.nodes.len(), 2);
    }

    #[test]
    fn test_mutate_add_connection_reenable() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(true);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &config);

        // Ensure it doesn't try to create this connection.
        let connection = super::ConnectionGene::new(2, 1, 0, 1.0);
        genome.connections.push(connection);

        // Disable the connection.
        genome.connections[0].enabled = false;
        genome.connections[1].enabled = false;

        genome.mutate_add_connection(&mut config, &mut rng);

        // TODO(orglofch): Check the connection is enabled.

        assert_eq!(genome.connections.len(), 2);
        assert_eq!(genome.nodes.len(), 2);
    }

    #[test]
    fn test_mutate_add_node() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(true);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &config);

        genome.mutate_add_node(&mut config, &mut rng);

        assert_eq!(genome.connections.len(), 3);
        assert_eq!(genome.nodes.len(), 3);
    }

    #[test]
    fn test_mutate_add_node_no_free_connection() {
        let mut config = super::GenomeConfig::new(0, 1);
        config.set_start_connected(false);

        let mut rng = super::rand::thread_rng();

        let mut genome = super::Genome::new(1, &config);

        genome.mutate_add_node(&mut config, &mut rng);

        assert_eq!(genome.connections.len(), 0);
        assert_eq!(genome.nodes.len(), 2);
    }
}
