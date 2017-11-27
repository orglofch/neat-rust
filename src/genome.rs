extern crate rand;

use self::rand::Rng;

use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use activation::ActivationFn;
use aggregation::AggregationFn;
use innovation::InnovationArchive;
use speciation::SpeciationConfig;

// TODO(orglofch): Consider separating initial and mutation configs.
/// The configuration controlling genome creation and mutation.
///
/// Note, the sum of mutation probability need not necessary add up to 1.
/// If `sum < 1` there wil be a chance of no mutation being taken.
/// If `sum > 1` the probability for a single mutation will be relative to the total.
pub struct GenomeConfig {
    // TODO(orglofch): Consider using tuples as keys instead of maintaing a mapping.
    innovation_archive: InnovationArchive,

    // TODO(orglofch): Allow inputs and outputs to be named, making the addition or removal
    // of input or outputs in checkpointing more explicit.
    /// The named inputs of a `Genome`.
    ///
    /// Note, this excludes bias as an input.
    inputs: Vec<String>,

    /// The named outputs of a `Genome`.
    outputs: Vec<String>,

    // TODO(orglofch): Allow for configuration that have multiple mutations
    // for every mutation iteration.
    /// Whether the input should start connected to the outputs.
    ///
    /// Note, in some circumstances having the outputs initially unconnected can
    /// improve the ability to prune unnecessary dependencies.
    start_connected: bool,

    /// Whether to allow recurrences to be formed.
    allow_recurrences: bool,

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
    pub fn new(inputs: Vec<String>, outputs: Vec<String>) -> GenomeConfig {
        GenomeConfig {
            innovation_archive: InnovationArchive::new(),
            inputs: inputs,
            outputs: outputs,
            start_connected: false,
            allow_recurrences: false,
            mutate_add_con_prob: 0.5,
            mutate_remove_con_prob: 0.0,
            mutate_add_node_prob: 0.5,
            mutate_remove_node_prob: 0.0,
        }
    }

    pub fn set_inputs(&mut self, inputs: Vec<String>) -> &mut GenomeConfig {
        self.inputs = inputs;
        self
    }

    pub fn set_outputs(&mut self, outputs: Vec<String>) -> &mut GenomeConfig {
        self.outputs = outputs;
        self
    }

    pub fn set_start_connected(&mut self, enabled: bool) -> &mut GenomeConfig {
        self.start_connected = enabled;
        self
    }

    pub fn set_allow_recurrences(&mut self, enabled: bool) -> &mut GenomeConfig {
        self.allow_recurrences = enabled;
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
pub struct NodeGene {
    /// The function used in aggregating across synapses prior to calculating activation.
    aggregation_fn: AggregationFn,

    /// The function used inaga determining neuron activation.
    activation_fn: ActivationFn,

    /// The bias applied to the node.
    ///
    /// We store the bias in the `NodeGene` instead of via a `ConnectionGene` since
    /// we don't want bias connection to be splittable and bias is generally fully connected
    /// or disabled (via `bias = 0.0`).
    bias: f32,
}

impl NodeGene {
    fn new(aggregation_fn: AggregationFn, activation_fn: ActivationFn) -> NodeGene {
        NodeGene {
            aggregation_fn: aggregation_fn,
            activation_fn: activation_fn,
            bias: 0.0,
        }
    }

    /// Calculates the compatibiltiy distance between this `NodeGene` and another.
    fn distance(&self, other: &NodeGene, speciation_config: &SpeciationConfig) -> f32 {
        return (self.bias - other.bias).abs() *
            speciation_config.compatibility_weight_coefficient;
    }
}

#[derive(Clone)]
pub struct ConnectionGene {
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
    fn new(weight: f32) -> ConnectionGene {
        ConnectionGene {
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

    /// `NodeGene` ids by their named inputs.
    input_ids_by_name: HashMap<String, u32>,

    /// `NodeGene` ids by their named outputs.
    /// TODO(orglofch): Consider using &strs.
    output_ids_by_name: HashMap<String, u32>,

    /// Output `NodeGenes` by their id.
    /// TODO(orglofch): Consider using &strs.
    output_nodes_by_id: HashMap<u32, NodeGene>,

    /// Hidden `NodeGenes` by their id.
    hidden_nodes_by_id: HashMap<u32, NodeGene>,

    /// `ConnectionGene` by their edge node ids.
    connections_by_edge: HashMap<(u32, u32), ConnectionGene>,
}

impl Genome {
    pub(crate) fn new(id: u32, config: &mut GenomeConfig) -> Genome {
        let mut input_ids_by_name: HashMap<String, u32> =
            HashMap::with_capacity(config.inputs.len());
        let mut output_ids_by_name: HashMap<String, u32> =
            HashMap::with_capacity(config.outputs.len());
        let mut output_nodes_by_id: HashMap<u32, NodeGene> =
            HashMap::with_capacity(config.outputs.len());

        let num_connections = if config.start_connected {
            config.inputs.len() * config.inputs.len()
        } else {
            0
        };

        let mut connections_by_edge: HashMap<(u32, u32), ConnectionGene> =
            HashMap::with_capacity(num_connections);

        // Create input nodes.
        for (i, name) in config.inputs.iter().enumerate() {
            let id = config.innovation_archive.record_spontaneous_input_node(i as u32);
            input_ids_by_name.insert(name.clone(), id);
        }

        // Create output nodes.
        for (i, name) in config.outputs.iter().enumerate() {
            let id = config.innovation_archive.record_spontaneous_output_node(i as u32);
            output_ids_by_name.insert(name.clone(), id);
            output_nodes_by_id.insert(id, NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid));
        }

        // If the inputs and outputs start connected then create the initial ConnectionGenes
        // which connect each input to each output.
        if config.start_connected {
            for &in_id in input_ids_by_name.values() {
                for &out_id in output_ids_by_name.values() {
                    let id = config.innovation_archive.record_connection_innovation(
                        in_id,
                        out_id,
                    );
                    connections_by_edge.insert((in_id, out_id), ConnectionGene::new(1.0));
                }
            }
        }

        Genome {
            id: id,
            input_ids_by_name: input_ids_by_name,
            output_ids_by_name: output_ids_by_name,
            output_nodes_by_id: output_nodes_by_id,
            hidden_nodes_by_id: HashMap::new(),
            connections_by_edge: connections_by_edge,
        }
    }

    /// Evaluates the `Genome` against the given input.
    ///
    /// # Arguments
    ///
    /// * `inputs` = The inputs to evaluate.
    pub (crate) fn activate(&self, inputs: &HashMap<String, f32>) -> HashMap<String, f32> {
        // TODO(orglofch): Allow fo parallel computation between layers.
        // TODO(orglofch): This only works for fead-forward NN's right now.

        // Gather inputs.
        let input_node_ids: HashSet<u32> = self.input_ids_by_name.values().cloned().collect();

        let mut out_ids_by_in_id: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut in_ids_by_out_id: HashMap<u32, HashSet<u32>> = HashMap::new();
        for edge in self.connections_by_edge.keys() {
            out_ids_by_in_id.entry(edge.0).or_insert(HashSet::new()).insert(edge.1);
            in_ids_by_out_id.entry(edge.1).or_insert(HashSet::new()).insert(edge.0);
        }

        // TODO(orglofch): Skip connections which don't connect to output.

        // Topological sort the elements.
        let mut activation_order: Vec<u32> = Vec::new();
        let mut active_inputs: Vec<u32> = input_node_ids.iter().cloned().collect();
        let mut remaining_edges_by_out_id: HashMap<u32, HashSet<u32>> = in_ids_by_out_id.clone();

        // TODO(orglofch): Add debug assertions to make sure we don't have cycles.
        while !active_inputs.is_empty() {
            let id = active_inputs.pop().unwrap();

            // Skip inputs in the activation order since they'll be populated manually.
            if !input_node_ids.contains(&id) {
                activation_order.push(id);
            }

            match out_ids_by_in_id.get(&id) {
                Some(ref out_ids) => {
                    for out_id in out_ids.iter() {
                        // Remove the connection from the list of in_ids.
                        match remaining_edges_by_out_id.get_mut(out_id) {
                            Some(ref mut in_ids) => {
                                in_ids.remove(&id);
                                if in_ids.is_empty() {
                                    active_inputs.push(*out_id);
                                }
                            }
                            None => panic!("Input ids and output ids are out of sync"),
                        }
                    }
                }
                None => (), // Node necessarily can't reach output.
            }
        }

        let mut outputs_by_id: HashMap<u32, f32> = HashMap::new();

        // Populate from the initial inputs.
        for (name, &value) in inputs {
            let id = self.input_ids_by_name.get(name).unwrap();
            outputs_by_id.insert(*id, value);
        }

        for id in activation_order {
            let mut values: Vec<f32> = Vec::new();

            for in_id in in_ids_by_out_id.get(&id).unwrap() {
                let value = *outputs_by_id.get(in_id).unwrap() * self.connections_by_edge.get(&(*in_id, id)).unwrap().weight;
                values.push(value);
            }

            let node = self.hidden_nodes_by_id.get(&id)
                .unwrap_or_else(|| self.output_nodes_by_id.get(&id).unwrap());

            // Include bias.
            values.push(node.bias);

            // Perform aggregation on the inputs.
            let aggregation = node.aggregation_fn.aggregate(values);

            let activation = node.activation_fn.eval(aggregation);

            outputs_by_id.insert(id, activation);
        }

        let mut outputs: HashMap<String, f32> = HashMap::new();
        for (name, id) in self.output_ids_by_name.iter() {
            // Outputs are not necessarily connected.
            match outputs_by_id.get(id) {
                Some(&value) => outputs.insert(name.clone(), value),
                None => outputs.insert(name.clone(), 0.0),
            };
        }

        return outputs;
    }

    /// Calculates the compatibility distance between this `Genome` and another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other `Genome` to compare against.
    ///
    /// * `speciation_config` - The configuration governing the equation used in calculating the distance.
    pub(crate) fn distance(&self, other: &Genome, speciation_config: &SpeciationConfig) -> f32 {
        // TODO(orglofch): Play around with other gene distance functions this currently treats excess
        // and disjoint genes as being equivalent.

        // NodeGene distance.
        let (disjoint_nodes, common_node_distance) = {
            let mut disjoint_genes = 0_u32;
            let mut common_distance = 0.0;

            // Hidden nodes.
            for (id, node) in self.hidden_nodes_by_id.iter() {
                match other.hidden_nodes_by_id.get(&id) {
                    Some(ref other_node) => {
                        common_distance += node.distance(other_node, speciation_config);
                    }
                    None => {
                        disjoint_genes += 1;
                    }
                }
            }

            for id in other.hidden_nodes_by_id.keys() {
                if !self.hidden_nodes_by_id.contains_key(&id) {
                    disjoint_genes += 1;
                }
            }

            // Output nodes.
            for (id, node) in self.output_nodes_by_id.iter() {
                let other_node = other.output_nodes_by_id.get(&id).unwrap();
                common_distance += node.distance(other_node, speciation_config);
            }
            (disjoint_genes, common_distance)
        };

        // ConnectionGene distance.
        let (disjoint_connections, common_connection_distance) = {
            let mut disjoint_genes = 0_u32;
            let mut common_distance = 0.0;

            for (edge, con) in self.connections_by_edge.iter() {
                match other.connections_by_edge.get(&edge) {
                    Some(ref other_con) => {
                        common_distance += con.distance(other_con, speciation_config);
                    }
                    None => {
                        disjoint_genes += 1;
                    }
                }
            }

            for edge in other.connections_by_edge.keys() {
                if !self.connections_by_edge.contains_key(&edge) {
                    disjoint_genes += 1;
                }
            }

            (disjoint_genes, common_distance)
        };

        let normalization_constant = {
            // TODO(orglofch): Maybe add other node types here.
            let self_gene_count = self.hidden_nodes_by_id.len() + self.connections_by_edge.len();
            let other_gene_count = other.hidden_nodes_by_id.len() + other.connections_by_edge.len();

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

        disjoint_distance + common_connection_distance + common_node_distance
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

    /// Mutates the `Genome` by adding a new `ConnectionGene` structural mutation.
    fn mutate_add_connection<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let active_connections: HashSet<(u32, u32)> = self.connections_by_edge.iter()
            .filter(|entry| entry.1.enabled)
            .map(|entry| *entry.0)
            .collect();

        let mut num_possible_inputs = self.input_ids_by_name.len() + self.hidden_nodes_by_id.len();
        if genome_config.allow_recurrences {
            num_possible_inputs += self.output_ids_by_name.len();
        }

        let num_possible_outputs = self.hidden_nodes_by_id.len() + self.output_ids_by_name.len();

        let mut possible_inputs: Vec<u32> = Vec::with_capacity(num_possible_inputs);
        let mut possible_outputs: Vec<u32> = Vec::with_capacity(num_possible_outputs);

        for &input_id in self.input_ids_by_name.values() {
            possible_inputs.push(input_id);
        }
        for &hidden_id in self.hidden_nodes_by_id.keys() {
            possible_inputs.push(hidden_id);
            possible_outputs.push(hidden_id);
        }
        for &output_id in self.output_ids_by_name.values() {
            if genome_config.allow_recurrences {
                possible_inputs.push(output_id);
            }
            possible_outputs.push(output_id);
        }

        let mut possible_connections: Vec<(u32, u32)> = Vec::new();
        for i in possible_inputs.into_iter() {
            for &o in possible_outputs.iter() {
                if active_connections.contains(&(i, o)) {
                    continue;
                }

                // TODO(orglofch): Unnecessarily expensive and temporary.
                if !genome_config.allow_recurrences && self.creates_cycle(i, o) {
                    continue;
                }

                possible_connections.push((i, o));
            }
        }

        if possible_connections.is_empty() {
            return;
        }

       let connection = rng.choose(&mut possible_connections).unwrap();

        // Check if the connection already exists but is disabled.
        {
            let maybe_existing_connection = self.connections_by_edge.get_mut(&connection);
            if maybe_existing_connection.is_some() {
                maybe_existing_connection.unwrap().enabled = true;
                return;
            }
        }

        // TODO(orglofch): Random weight?
        self.connections_by_edge.insert(*connection, ConnectionGene::new(1.0));
    }

    /// Mutate the `Genome` by adding a new `NodeGene` structural mutation.
    fn mutate_add_node<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let maybe_split_edge = self.split_random_connection(rng);
        if maybe_split_edge.is_none() {
            return;
        }

        let split_edge = maybe_split_edge.unwrap();

        // Retrieve the split connection.
        let con = self.connections_by_edge.get(&split_edge).unwrap().clone();

        // Create a new hidden node.
        let new_id = genome_config
            .innovation_archive
            .record_hidden_node_innovation(split_edge.0, split_edge.1, con.splits);
        self.hidden_nodes_by_id.insert(new_id, NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid));

        // Create two new connections.
        // The first connection is initialized with a weight of 1.
        // The second connection is initilized with the original connections weight.
        // This preserves the original phenotype.
        self.connections_by_edge.insert((split_edge.0, new_id), ConnectionGene::new(1.0));
        self.connections_by_edge.insert((new_id, split_edge.1), ConnectionGene::new(con.weight));
    }

    /// Mutate the `Genome` by modifying the weight on an existing `NodeGene`.
    fn mutate_weight<R: Rng>(&mut self, rng: &mut R) {
        // Select a random active connection.
        let mut active_connections: Vec<&mut ConnectionGene> = self.connections_by_edge
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

    /// Splits a random enabled connection and returns the edge it split.
    fn split_random_connection<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, u32)> {
        let mut splittable_edges: Vec<(u32, u32)> = self.connections_by_edge
            .iter_mut()
            .filter(|entry| entry.1.enabled)
            .map(|entry| *entry.0)
            .collect();
        if splittable_edges.is_empty() {
            return None;
        }

        let edge = rng.choose_mut(&mut splittable_edges).unwrap();

        let mut connection = self.connections_by_edge.get_mut(&edge).unwrap();

        // Disable the original connection.
        connection.enabled = false;

        // Increase the number of splits.
        connection.splits += 1;

        return Some(edge.clone());
    }

    /// Checks whether the addition of a connection would form a cycle.
    ///
    /// TODO(orglofch): Add tests for this.
    fn creates_cycle(&self, in_id: u32, out_id: u32) -> bool {
        // TODO(orglofch): We shouldn't include disabled connections but we need to
        // in the event they become enabled in the future.

        // Try to find an existing path (out_id -> in_id).
        let mut out_ids_by_in_id: HashMap<u32, HashSet<u32>> = HashMap::new();
        for edge in self.connections_by_edge.keys() {
            out_ids_by_in_id.entry(edge.0).or_insert(HashSet::new()).insert(edge.1);
        }

        // BFS to find the input id.
        let mut queue: VecDeque<u32> = VecDeque::new();
        queue.push_back(out_id);

        while !queue.is_empty() {
            match out_ids_by_in_id.get(&queue.pop_front().unwrap()) {
                Some(ref out_ids) => {
                    if out_ids.contains(&in_id) {
                        return true;
                    }
                    for out_id in out_ids.into_iter() {
                        queue.push_back(*out_id);
                    }
                }
                None => (),
            }
        }

        false
    }
}

pub type Population = Vec<Genome>;

// TODO(orglofch): More comprehensive equality tests.
// TODO(orglofch): Figure out how to force seed the rng without implementing the whole trait.
// TODO(orglofch): Adds tests for mutate weights.
#[cfg(test)]
mod test {
    use super::*;

    // TODO(orglofch): Add to macros.rs.
    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => ({
            let (a, b) = (&$a, &$b);
            assert!((*a - *b).abs() < 1.0e-6,
                    "{} is not approximately equal to {}", *a, *b);
        })
    }

    #[test]
    fn test_new_genome_unconnected() {
        let mut inputs = vec!["input_1".to_owned()];
        let mut outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs.clone(), outputs.clone());

        let genome = Genome::new(1, &mut genome_config);

        assert_eq!(genome.id, 1);
        assert_eq!(genome.connections_by_edge.len(), 0);
        assert_eq!(genome.input_ids_by_name.len(), 1);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);

        inputs.push("input_2".to_owned());
        genome_config.set_inputs(inputs);

        let genome = Genome::new(2, &mut genome_config);

        assert_eq!(genome.id, 2);
        assert_eq!(genome.connections_by_edge.len(), 0);
        assert_eq!(genome.input_ids_by_name.len(), 2);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);

        outputs.push("output_2".to_owned());
        genome_config.set_outputs(outputs);

        let genome = Genome::new(3, &mut genome_config);

        assert_eq!(genome.id, 3);
        assert_eq!(genome.connections_by_edge.len(), 0);
        assert_eq!(genome.input_ids_by_name.len(), 2);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 2);
    }

    #[test]
    fn test_new_genome_start_connected() {
        let mut inputs = vec!["input_1".to_owned()];
        let mut outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs.clone(), outputs.clone());
        genome_config.set_start_connected(true);

        let genome = Genome::new(1, &mut genome_config);

        assert_eq!(genome.id, 1);
        assert_eq!(genome.connections_by_edge.len(), 1);
        assert_eq!(genome.input_ids_by_name.len(), 1);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);

        inputs.push("input_2".to_owned());
        genome_config.set_inputs(inputs);

        let genome = Genome::new(2, &mut genome_config);

        assert_eq!(genome.id, 2);
        assert_eq!(genome.connections_by_edge.len(), 2);
        assert_eq!(genome.input_ids_by_name.len(), 2);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);

        outputs.push("output_2".to_owned());
        genome_config.set_outputs(outputs);

        let genome = Genome::new(3, &mut genome_config);

        assert_eq!(genome.id, 3);
        assert_eq!(genome.connections_by_edge.len(), 4);
        assert_eq!(genome.input_ids_by_name.len(), 2);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 2);
    }

    #[test]
    fn test_activate_unconnected() {
        let output = "output".to_owned();
        let input = "input".to_owned();

        let inputs = vec![input.clone()];
        let outputs = vec![output.clone()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(false);

        let genome = Genome::new(0, &mut genome_config);

        let mut inputs: HashMap<String, f32> = HashMap::new();
        inputs.insert(input.clone(), 1.0);

        let results = genome.activate(&inputs);

        assert_approx_eq!(results.get(&output).unwrap(), 0.0);
    }

    #[test]
    fn test_activate_xor() {
        let output = "output_1".to_owned();
        let input_1 = "input_1".to_owned();
        let input_2 = "input_2".to_owned();

        let inputs = vec![input_1.clone(), input_2.clone()];
        let outputs = vec![output.clone()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(false);

        let mut genome = Genome::new(0, &mut genome_config);

        genome.output_nodes_by_id.values_mut().next().unwrap().bias = -300.0;

        genome.hidden_nodes_by_id.insert(
            3,
            NodeGene::new(
                AggregationFn::Sum,
                ActivationFn::Sigmoid,
            ),
        );
        genome.hidden_nodes_by_id.get_mut(&3).unwrap().bias = -100.0;

        genome.hidden_nodes_by_id.insert(
            4,
            NodeGene::new(
                AggregationFn::Sum,
                ActivationFn::Sigmoid,
            ),
        );
        genome.hidden_nodes_by_id.get_mut(&4).unwrap().bias = 300.0;

        // TODO(orglofch): This assume input vs output order.
        genome.connections_by_edge.insert((0, 3), ConnectionGene::new(200.0));
        genome.connections_by_edge.insert((1, 3), ConnectionGene::new(200.0));

        genome.connections_by_edge.insert((0, 4), ConnectionGene::new(-200.0));
        genome.connections_by_edge.insert((1, 4), ConnectionGene::new(-200.0));

        genome.connections_by_edge.insert((3, 2), ConnectionGene::new(200.0));
        genome.connections_by_edge.insert((4, 2), ConnectionGene::new(200.0));

        // (1, 1) => 0.
        let mut inputs: HashMap<String, f32> = HashMap::new();
        inputs.insert(input_1.clone(), 1.0);
        inputs.insert(input_2.clone(), 1.0);

        let results = genome.activate(&inputs);

        assert_approx_eq!(results.get(&output).unwrap(), 0.0);

        // (1, 0) => 1.
        let mut inputs: HashMap<String, f32> = HashMap::new();
        inputs.insert(input_1.clone(), 1.0);
        inputs.insert(input_2.clone(), 0.0);

        let results = genome.activate(&inputs);

        assert_approx_eq!(results.get(&output).unwrap(), 1.0);

        // (0, 1) => 1.
        let mut inputs: HashMap<String, f32> = HashMap::new();
        inputs.insert(input_1.clone(), 0.0);
        inputs.insert(input_2.clone(), 1.0);

        let results = genome.activate(&inputs);

        assert_approx_eq!(results.get(&output).unwrap(), 1.0);

        // (0, 0) => 0.
        let mut inputs: HashMap<String, f32> = HashMap::new();
        inputs.insert(input_1.clone(), 0.0);
        inputs.insert(input_2.clone(), 0.0);

        let results = genome.activate(&inputs);

        assert_approx_eq!(results.get(&output).unwrap(), 0.0);
    }

    #[test]
    fn test_distance_node() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);

        let mut speciation_config = SpeciationConfig::new();
        speciation_config.set_compatibility_weight_coefficient(1.0);

        let mut genome_1 = Genome::new(0, &mut genome_config);
        let mut genome_2 = Genome::new(0, &mut genome_config);

        genome_1.output_nodes_by_id.values_mut().next().unwrap().bias = 1.0;
        genome_2.output_nodes_by_id.values_mut().next().unwrap().bias = 1.0;

        // Same bias.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);

        genome_1.output_nodes_by_id.values_mut().next().unwrap().bias = 0.5;

        // Bias differs.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.5);

        speciation_config.set_compatibility_weight_coefficient(0.5);

        // Bias differ and coefficient is fractional.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.25);

        genome_2.output_nodes_by_id.values_mut().next().unwrap().bias = 0.5;

        // Bias is the same but coefficient is fractional.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);
    }

    #[test]
    fn test_distance_connection() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(true);

        let mut speciation_config = SpeciationConfig::new();
        speciation_config.set_compatibility_weight_coefficient(1.0);

        let mut genome_1 = Genome::new(0, &mut genome_config);
        let mut genome_2 = Genome::new(0, &mut genome_config);

        genome_1.connections_by_edge.get_mut(&(0, 1)).unwrap().weight = 1.0;
        genome_2.connections_by_edge.get_mut(&(0, 1)).unwrap().weight = 1.0;

        // Same weights.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);

        genome_1.connections_by_edge.get_mut(&(0, 1)).unwrap().weight = 0.5;

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

        genome_2.connections_by_edge.get_mut(&(0, 1)).unwrap().weight = 0.5;

        // Weights are the same but coefficient is fractional.
        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);
    }

    #[test]
    fn test_distance_disjoint() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);

        let mut speciation_config = SpeciationConfig::new();
        speciation_config.set_compatibility_disjoint_coefficient(1.0);

        let mut genome_1 = Genome::new(0, &mut genome_config);
        let mut genome_2 = Genome::new(1, &mut genome_config);

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.0);

        // Nodes are disjoint.
        genome_1.hidden_nodes_by_id.insert(
            2,
            NodeGene::new(
                AggregationFn::Sum,
                ActivationFn::Sigmoid,
            ),
        );

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 1.0);

        // Nodes and connections are disjoint.
        genome_2.connections_by_edge.insert((1, 2), ConnectionGene::new(1.0));

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
        genome_2.hidden_nodes_by_id.insert(
            2,
            NodeGene::new(
                AggregationFn::Sum,
                ActivationFn::Sigmoid,
            ),
        );

        assert_eq!(
            genome_1.distance(&genome_2, &speciation_config),
            genome_2.distance(&genome_1, &speciation_config)
        );
        assert_eq!(genome_1.distance(&genome_2, &speciation_config), 0.5);
    }

    #[test]
    fn test_mutate_add_connection_new() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(false);

        let mut rng = rand::thread_rng();

        let mut genome = Genome::new(1, &mut genome_config);

        genome.mutate_add_connection(&mut genome_config, &mut rng);

        assert_eq!(genome.connections_by_edge.len(), 1);
        assert_eq!(genome.input_ids_by_name.len(), 1);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);
    }

    #[test]
    fn test_mutate_add_connection_no_free_connection() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(true)
            .set_allow_recurrences(false);

        let mut rng = rand::thread_rng();

        let mut genome = Genome::new(1, &mut genome_config);

        genome.mutate_add_connection(&mut genome_config, &mut rng);

        assert_eq!(genome.connections_by_edge.len(), 1);
        assert_eq!(genome.input_ids_by_name.len(), 1);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);
    }

    // TODO(orglofch): Test with recurrences enabled.

    // TODO(orglofch): fn test_mutate_add_connection_reenable() {}

    #[test]
    fn test_mutate_add_node() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(true);

        let mut rng = rand::thread_rng();

        let mut genome = Genome::new(1, &mut genome_config);

        genome.mutate_add_node(&mut genome_config, &mut rng);

        assert_eq!(genome.connections_by_edge.len(), 3);
        assert_eq!(genome.input_ids_by_name.len(), 1);
        assert_eq!(genome.hidden_nodes_by_id.len(), 1);
        assert_eq!(genome.output_ids_by_name.len(), 1);
    }

    #[test]
    fn test_mutate_add_node_no_free_connection() {
        let inputs = vec!["input_1".to_owned()];
        let outputs = vec!["output_1".to_owned()];
        let mut genome_config = GenomeConfig::new(inputs, outputs);
        genome_config.set_start_connected(false);

        let mut rng = rand::thread_rng();

        let mut genome = Genome::new(1, &mut genome_config);

        genome.mutate_add_node(&mut genome_config, &mut rng);

        assert_eq!(genome.connections_by_edge.len(), 0);
        assert_eq!(genome.input_ids_by_name.len(), 1);
        assert_eq!(genome.hidden_nodes_by_id.len(), 0);
        assert_eq!(genome.output_ids_by_name.len(), 1);
    }
}
