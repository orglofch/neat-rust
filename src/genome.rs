extern crate rand;

use self::rand::Rng;

use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use activation::ActivationFn;
use aggregation::AggregationFn;
use innovation::InnovationArchive;
use gene::{ConnectionGene, NodeGene};
use speciation::SpeciationConfig;

// TODO(orglofch): Consider separating initial and mutatable configs.
/// The configuration controlling genome creation and mutation.
///
/// Note, the sum of mutation probability need not necessary add up to 1.
/// If `sum < 1` there wil be a chance of no mutation being taken.
/// If `sum > 1` the probability for a single mutation will be relative to the total.
pub struct GenomeConfig<'a> {
    innovation_archive: InnovationArchive,

    // TODO(orglofch): Allow inputs and outputs to be named, making the addition or removal
    // of input or outputs in checkpointing more explicit.
    /// The named inputs of a `Genome`.
    ///
    /// Note, this excludes bias as an input.
    inputs: Vec<&'a str>,

    /// The named outputs of a `Genome`.
    outputs: Vec<&'a str>,

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

impl<'a> GenomeConfig<'a> {
    pub fn new(inputs: Vec<&'a str>, outputs: Vec<&'a str>) -> GenomeConfig<'a> {
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

    pub fn set_inputs(&mut self, inputs: Vec<&'a str>) -> &mut GenomeConfig<'a> {
        self.inputs = inputs;
        self
    }

    pub fn set_outputs(&mut self, outputs: Vec<&'a str>) -> &mut GenomeConfig<'a> {
        self.outputs = outputs;
        self
    }

    pub fn set_start_connected(&mut self, enabled: bool) -> &mut GenomeConfig<'a> {
        self.start_connected = enabled;
        self
    }

    pub fn set_allow_recurrences(&mut self, enabled: bool) -> &mut GenomeConfig<'a> {
        self.allow_recurrences = enabled;
        self
    }

    pub fn set_mutate_add_connection_probability(&mut self, probability: f32) -> &mut GenomeConfig<'a> {
        self.mutate_add_con_prob = probability;
        self
    }

    pub fn set_mutate_remove_connection_probability(&mut self, probability: f32) -> &mut GenomeConfig<'a> {
        self.mutate_remove_con_prob = probability;
        self
    }

    pub fn set_mutate_add_node_probability(&mut self, probability: f32) -> &mut GenomeConfig<'a> {
        self.mutate_add_node_prob = probability;
        self
    }

    pub fn set_mutate_remove_node_probability(&mut self, probability: f32) -> &mut GenomeConfig<'a> {
        self.mutate_remove_node_prob = probability;
        self
    }
}

#[derive(Debug)]
pub struct Genome<'a> {
    /// `NodeGene` ids by their named inputs.
    input_ids_by_name: HashMap<&'a str, u32>,

    /// `NodeGene` ids by their named outputs.
    /// TODO(orglofch): Consider using &strs.
    output_ids_by_name: HashMap<&'a str, u32>,

    /// Output `NodeGenes` by their id.
    /// TODO(orglofch): Consider using &strs.
    output_nodes_by_id: HashMap<u32, NodeGene>,

    /// Hidden `NodeGenes` by their id.
    hidden_nodes_by_id: HashMap<u32, NodeGene>,

    /// `ConnectionGene` by their edge node ids.
    connections_by_edge: HashMap<(u32, u32), ConnectionGene>,
}

impl<'a> Genome<'a> {
    // TODO(orglofch): Try to make GenomeConfig immutable if we can factor out the innovation archive.
    pub(crate) fn new(genome_config: &mut GenomeConfig<'a>) -> Genome<'a> {
        let mut input_ids_by_name: HashMap<&str, u32> = HashMap::with_capacity(genome_config.inputs.len());
        let mut output_ids_by_name: HashMap<&str, u32> = HashMap::with_capacity(genome_config.outputs.len());
        let mut output_nodes_by_id: HashMap<u32, NodeGene> = HashMap::with_capacity(genome_config.outputs.len());

        let num_connections = if genome_config.start_connected {
            genome_config.inputs.len() * genome_config.inputs.len()
        } else {
            0
        };

        let mut connections_by_edge: HashMap<(u32, u32), ConnectionGene> = HashMap::with_capacity(num_connections);

        // Create input nodes.
        for (i, name) in genome_config.inputs.iter().enumerate() {
            let id = genome_config
                .innovation_archive
                .record_spontaneous_input_node(i as u32);
            input_ids_by_name.insert(name, id);
        }

        // Create output nodes.
        for (i, name) in genome_config.outputs.iter().enumerate() {
            let id = genome_config
                .innovation_archive
                .record_spontaneous_output_node(i as u32);
            output_ids_by_name.insert(name, id);
            output_nodes_by_id.insert(id, NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid));
        }

        // If the inputs and outputs start connected then create the initial ConnectionGenes
        // which connect each input to each output.
        if genome_config.start_connected {
            for &in_id in input_ids_by_name.values() {
                for &out_id in output_ids_by_name.values() {
                    connections_by_edge.insert((in_id, out_id), ConnectionGene::new(1.0));
                }
            }
        }

        Genome {
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
    pub(crate) fn activate(&self, inputs: &HashMap<&'a str, f32>) -> HashMap<&'a str, f32> {
        // TODO(orglofch): Allow fo parallel computation between layers.
        // TODO(orglofch): This only works for fead-forward NN's right now.

        // Gather inputs.
        let input_node_ids: HashSet<u32> = self.input_ids_by_name.values().cloned().collect();

        let mut out_ids_by_in_id: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut in_ids_by_out_id: HashMap<u32, HashSet<u32>> = HashMap::new();
        for edge in self.connections_by_edge.keys() {
            out_ids_by_in_id
                .entry(edge.0)
                .or_insert(HashSet::new())
                .insert(edge.1);
            in_ids_by_out_id
                .entry(edge.1)
                .or_insert(HashSet::new())
                .insert(edge.0);
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
                let value = self.connections_by_edge
                    .get(&(*in_id, id))
                    .unwrap()
                    .activate(*outputs_by_id.get(in_id).unwrap());
                values.push(value);
            }

            let node = self.hidden_nodes_by_id.get(&id).unwrap_or_else(|| {
                self.output_nodes_by_id.get(&id).unwrap()
            });

            outputs_by_id.insert(id, node.activate(&values));
        }

        let mut outputs: HashMap<&'a str, f32> = HashMap::new();
        for (name, id) in self.output_ids_by_name.iter() {
            // Outputs are not necessarily connected.
            match outputs_by_id.get(id) {
                Some(&value) => outputs.insert(name, value),
                None => outputs.insert(name, 0.0),
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
                    None => disjoint_genes += 1,
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
                    None => disjoint_genes += 1,
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

    /// Create a new genome via crossover between this `Genome` and another.
    ///
    /// TODO(orglofch): Remove the assumption that self is the more fit of the two genomes.
    pub(crate) fn crossover(&self, other: &Genome, genome_config: &mut GenomeConfig) -> Genome {
        // TODO(orglofch): Consider validating that they are exactly the same in terms of names as well.
        debug_assert_eq!(self.input_ids_by_name.len(), other.input_ids_by_name.len());
        debug_assert_eq!(
            self.output_nodes_by_id.len(),
            other.output_nodes_by_id.len()
        );

        // TODO(orglofch): Keep the disjoint entries from the more fit parent.
        // TODO(orglofch): Clone the more fit parent and mutate it.
        let genome = self.clone();
        let less_fit_genome = other;

        // TODO(orglofch): Consider relying on the config here.
        let mut hidden_nodes_by_id: HashMap<u32, NodeGene> = HashMap::with_capacity(self.hidden_nodes_by_id.len());
        let mut output_nodes_by_id: HashMap<u32, NodeGene> = HashMap::with_capacity(genome_config.outputs.len());
        let mut connections_by_edge: HashMap<(u32, u32), ConnectionGene> =
            HashMap::with_capacity(self.connections_by_edge.len());

        // Hidden nodes.
        for (id, node) in genome.hidden_nodes_by_id.iter() {
            match less_fit_genome.hidden_nodes_by_id.get(&id) {
                Some(ref other_node) => {
                    hidden_nodes_by_id.insert(*id, node.crossover(other_node));
                }
                None => (),
            }
        }

        // Output nodes.
        for (id, node) in genome.output_nodes_by_id.iter() {
            match less_fit_genome.hidden_nodes_by_id.get(&id) {
                Some(ref other_node) => {
                    output_nodes_by_id.insert(*id, node.crossover(other_node));
                }
                None => (),
            }
        }

        // Connections.
        for (edge, con) in genome.connections_by_edge.iter() {
            match less_fit_genome.connections_by_edge.get(&edge) {
                Some(ref other_con) => {
                    connections_by_edge.insert(*edge, con.crossover(other_con));
                }
                None => (),
            }
        }

        Genome {
            input_ids_by_name: self.input_ids_by_name.clone(),
            output_ids_by_name: self.output_ids_by_name.clone(),
            output_nodes_by_id: output_nodes_by_id,
            hidden_nodes_by_id: hidden_nodes_by_id,
            connections_by_edge: connections_by_edge,
        }
    }

    /// Mutates the `Genome`.
    ///
    /// # Arguments
    ///
    /// * `genome_config` - The configuration governing the mutation of the `Genome`.
    pub(crate) fn mutate<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let sum_prob = genome_config.mutate_add_con_prob + genome_config.mutate_remove_con_prob +
            genome_config.mutate_add_node_prob + genome_config.mutate_remove_node_prob;

        // If the sum of the probabilities is greater than 1, normalize the probability range
        // so the sum consistutes a 100% probability. Otherwise, allow the total probability
        // to be less than 1 to allow for a probability of no mutations.
        let mut prob = rng.gen_range::<f32>(0.0, sum_prob.max(1.0));

        if prob < genome_config.mutate_add_con_prob {
            self.mutate_add_connection(genome_config, rng);
            return;
        }
        prob -= genome_config.mutate_add_con_prob;

        if prob < genome_config.mutate_remove_con_prob {
            // TODO(orglofch): Implement.
            return;
        }
        prob -= genome_config.mutate_remove_con_prob;

        if prob < genome_config.mutate_add_node_prob {
            self.mutate_add_node(genome_config, rng);
            return;
        }
        prob -= genome_config.mutate_add_node_prob;

        if prob < genome_config.mutate_remove_node_prob {
            // TODO(orglofch): Implement.
            return;
        }

        self.mutate_weight(rng);
    }

    /// Mutates the `Genome` by adding a new `ConnectionGene` structural mutation.
    fn mutate_add_connection<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let active_connections: HashSet<(u32, u32)> = self.connections_by_edge
            .iter()
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
        self.connections_by_edge.insert(
            *connection,
            ConnectionGene::new(1.0),
        );
    }

    /// Mutate the `Genome` by removing a `ConnectionGene` structural mutation.
    fn mutate_remove_connection<R: Rng>(&mut self, rng: &mut R) {
        let mut active_connections: Vec<&mut ConnectionGene> = self.connections_by_edge
            .iter_mut()
            .map(|entry| entry.1)
            .filter(|con| con.enabled)
            .collect();
        if active_connections.is_empty() {
            return;
        }

        let con = rng.choose_mut(&mut active_connections).unwrap();

        // We don't actually remove connections since they're important for speciation,
        // instead we disable the connection.
        con.enabled = false;
    }

    /// Mutate the `Genome` by adding a new `NodeGene` structural mutation.
    fn mutate_add_node<R: Rng>(&mut self, genome_config: &mut GenomeConfig, rng: &mut R) {
        let mut splittable_edges: Vec<(u32, u32)> = self.connections_by_edge
            .iter_mut()
            .filter(|entry| entry.1.enabled)
            .map(|entry| *entry.0)
            .collect();
        if splittable_edges.is_empty() {
            return;
        }

        let split_edge = rng.choose_mut(&mut splittable_edges).unwrap();

        // Retrieve the split connection.
        let con = self.connections_by_edge.get(&split_edge).unwrap().clone();

        // Create a new hidden node.
        let new_id = genome_config
            .innovation_archive
            .record_hidden_node_innovation(split_edge.0, split_edge.1, con.splits);
        self.hidden_nodes_by_id.insert(
            new_id,
            NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid),
        );

        // Update the previous connection.
        {
            let connection = self.connections_by_edge.get_mut(&split_edge).unwrap();

            // Disable the original connection.
            connection.enabled = false;

            // Increase the number of splits.
            connection.splits += 1;
        }

        // Create two new connections.
        // The first connection is initialized with a weight of 1.
        // The second connection is initilized with the original connections weight.
        // This preserves the original phenotype.
        self.connections_by_edge.insert(
            (split_edge.0, new_id),
            ConnectionGene::new(1.0),
        );
        self.connections_by_edge.insert(
            (new_id, split_edge.1),
            ConnectionGene::new(con.weight),
        );
    }

    /// Mutate the `Genome` by removing a `NodeGene` structural mutation.
    fn mutate_remove_node<R: Rng>(&mut self, rng: &mut R) {
        // TODO(orglofch): Implement.
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

        let connection = rng.choose_mut(&mut active_connections).unwrap();

        // TODO(orglofch): Make the distribution variable, possibly based on a covariance.
        connection.weight += rng.gen_range::<f32>(-1.0, 1.0);
    }

    /// Checks whether the addition of a connection would form a cycle.
    fn creates_cycle(&self, in_id: u32, out_id: u32) -> bool {
        // TODO(orglofch): We shouldn't include disabled connections but we need to
        // in the event they become enabled in the future.
        if in_id == out_id {
            return true;
        }

        // Try to find an existing path (out_id -> in_id).
        let mut out_ids_by_in_id: HashMap<u32, HashSet<u32>> = HashMap::new();
        for edge in self.connections_by_edge.keys() {
            out_ids_by_in_id
                .entry(edge.0)
                .or_insert(HashSet::new())
                .insert(edge.1);
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

/// A collection of `Genomes` by id.
pub type Population<'a> = HashMap<u32, Genome<'a>>;

// TODO(orglofch): More comprehensive equality tests.
// TODO(orglofch): Figure out how to force seed the rng without implementing the whole trait.
// TODO(orglofch): Adds tests for mutate weights.
#[cfg(test)]
mod test {
    use super::*;

    use super::rand::rngs::mock::StepRng;

    const INPUT_1: &'static str = "input_1";
    const INPUT_2: &'static str = "input_2";
    const OUTPUT_1: &'static str = "output_1";
    const OUTPUT_2: &'static str = "output_2";

    #[test]
    fn test_new_genome_unconnected() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.connections_by_edge.len(), 0);
        assert_eq!(gen.input_ids_by_name.len(), 1);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);

        gen_conf.set_inputs(vec![INPUT_1, INPUT_2]);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.connections_by_edge.len(), 0);
        assert_eq!(gen.input_ids_by_name.len(), 2);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);

        gen_conf.set_outputs(vec![OUTPUT_1, OUTPUT_2]);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.connections_by_edge.len(), 0);
        assert_eq!(gen.input_ids_by_name.len(), 2);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 2);
    }

    #[test]
    fn test_new_genome_start_connected() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.connections_by_edge.len(), 1);
        assert_eq!(gen.input_ids_by_name.len(), 1);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);

        gen_conf.set_inputs(vec![INPUT_1, INPUT_2]);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.connections_by_edge.len(), 2);
        assert_eq!(gen.input_ids_by_name.len(), 2);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);

        gen_conf.set_outputs(vec![OUTPUT_1, OUTPUT_2]);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.connections_by_edge.len(), 4);
        assert_eq!(gen.input_ids_by_name.len(), 2);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 2);
    }

    #[test]
    fn test_activate_unconnected() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(false);

        let gen = Genome::new(&mut gen_conf);

        let inputs: HashMap<&str, f32> = hashmap! { INPUT_1 => 1.0 };

        let results = gen.activate(&inputs);

        assert_approx_eq!(results.get(&OUTPUT_1).unwrap(), 0.0);
    }

    #[test]
    fn test_activate_xor() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1, INPUT_2], vec![OUTPUT_1]);
        gen_conf.set_start_connected(false);

        let mut gen = Genome::new(&mut gen_conf);

        gen.output_nodes_by_id.values_mut().next().unwrap().bias = -300.0;

        let hidden_1 = NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid);
        gen.hidden_nodes_by_id.insert(3, hidden_1);
        gen.hidden_nodes_by_id.get_mut(&3).unwrap().bias = -100.0;

        let hidden_2 = NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid);
        gen.hidden_nodes_by_id.insert(4, hidden_2);
        gen.hidden_nodes_by_id.get_mut(&4).unwrap().bias = 300.0;

        // TODO(orglofch): This assume input vs output order.
        gen.connections_by_edge.insert(
            (0, 3),
            ConnectionGene::new(200.0),
        );
        gen.connections_by_edge.insert(
            (1, 3),
            ConnectionGene::new(200.0),
        );

        gen.connections_by_edge.insert(
            (0, 4),
            ConnectionGene::new(-200.0),
        );
        gen.connections_by_edge.insert(
            (1, 4),
            ConnectionGene::new(-200.0),
        );

        gen.connections_by_edge.insert(
            (3, 2),
            ConnectionGene::new(200.0),
        );
        gen.connections_by_edge.insert(
            (4, 2),
            ConnectionGene::new(200.0),
        );

        // (1, 1) => 0.
        let inputs: HashMap<&str, f32> = hashmap! {
            INPUT_1 => 1.0,
            INPUT_2 => 1.0
        };

        let results = gen.activate(&inputs);

        assert_approx_eq!(results.get(&OUTPUT_1).unwrap(), 0.0);

        // (1, 0) => 1.
        let inputs: HashMap<&str, f32> = hashmap! {
            INPUT_1 => 1.0,
            INPUT_2 => 0.0
        };

        let results = gen.activate(&inputs);

        assert_approx_eq!(results.get(&OUTPUT_1).unwrap(), 1.0);

        // (0, 1) => 1.
        let inputs: HashMap<&str, f32> = hashmap! {
            INPUT_1 => 0.0,
            INPUT_2 => 1.0
        };

        let results = gen.activate(&inputs);

        assert_approx_eq!(results.get(&OUTPUT_1).unwrap(), 1.0);

        // (0, 0) => 0.
        let inputs: HashMap<&str, f32> = hashmap! {
            INPUT_1 => 0.0,
            INPUT_2 => 0.0
        };

        let results = gen.activate(&inputs);

        assert_approx_eq!(results.get(&OUTPUT_1).unwrap(), 0.0);
    }

    #[test]
    fn test_distance_node() {
        // TODO(orglofch): Implement.
    }

    #[test]
    fn test_distance_connection() {
        // TODO(orglofch): Implement.z
    }

    #[test]
    fn test_distance_disjoint() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let mut spec_conf = SpeciationConfig::new();
        spec_conf.set_compatibility_disjoint_coefficient(1.0);

        let mut gen_1 = Genome::new(&mut gen_conf);
        let mut gen_2 = Genome::new(&mut gen_conf);

        assert_eq!(
            gen_1.distance(&gen_2, &spec_conf),
            gen_2.distance(&gen_1, &spec_conf)
        );
        assert_eq!(gen_1.distance(&gen_2, &spec_conf), 0.0);

        // Nodes are disjoint.
        let node = NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid);
        gen_1.hidden_nodes_by_id.insert(2, node);

        assert_eq!(
            gen_1.distance(&gen_2, &spec_conf),
            gen_2.distance(&gen_1, &spec_conf)
        );
        assert_eq!(gen_1.distance(&gen_2, &spec_conf), 1.0);

        // Nodes and connections are disjoint.
        gen_2.connections_by_edge.insert(
            (1, 2),
            ConnectionGene::new(1.0),
        );

        assert_eq!(
            gen_1.distance(&gen_2, &spec_conf),
            gen_2.distance(&gen_1, &spec_conf)
        );
        assert_eq!(gen_1.distance(&gen_2, &spec_conf), 2.0);

        // Nodes and connections are disjoint and coefficient is fractional.
        spec_conf.set_compatibility_disjoint_coefficient(0.5);

        assert_eq!(
            gen_1.distance(&gen_2, &spec_conf),
            gen_2.distance(&gen_1, &spec_conf)
        );
        assert_eq!(gen_1.distance(&gen_2, &spec_conf), 1.0);

        // Connections are disjoint.
        let node = NodeGene::new(AggregationFn::Sum, ActivationFn::Sigmoid);
        gen_2.hidden_nodes_by_id.insert(2, node);

        assert_eq!(
            gen_1.distance(&gen_2, &spec_conf),
            gen_2.distance(&gen_1, &spec_conf)
        );
        assert_eq!(gen_1.distance(&gen_2, &spec_conf), 0.5);
    }

    #[test]
    fn test_crossover() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let gen_1 = Genome::new(&mut gen_conf);
        let gen_2 = Genome::new(&mut gen_conf);

        let _ = gen_1.crossover(&gen_2, &mut gen_conf);

        // TODO(orglofch):
    }

    #[test]
    #[should_panic]
    fn test_crossover_incompatible_inputs() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let gen_1 = Genome::new(&mut gen_conf);

        // Change the number of inputs.
        gen_conf = GenomeConfig::new(vec![INPUT_1, INPUT_2], vec![OUTPUT_1]);

        let gen_2 = Genome::new(&mut gen_conf);

        gen_1.crossover(&gen_2, &mut gen_conf);
    }

    #[test]
    #[should_panic]
    fn test_crossover_incompatible_outputs() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let gen_1 = Genome::new(&mut gen_conf);

        // Change the number of outputs.
        gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1, OUTPUT_2]);

        let gen_2 = Genome::new(&mut gen_conf);

        gen_1.crossover(&gen_2, &mut gen_conf);
    }

    #[test]
    fn test_mutate_add_connection_new() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(false);

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        gen.mutate_add_connection(&mut gen_conf, &mut rng);

        assert_eq!(gen.connections_by_edge.len(), 1);
        assert_eq!(gen.input_ids_by_name.len(), 1);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);
    }

    #[test]
    fn test_mutate_add_connection_no_free_connection() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true).set_allow_recurrences(
            false,
        );

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        gen.mutate_add_connection(&mut gen_conf, &mut rng);

        assert_eq!(gen.connections_by_edge.len(), 1);
        assert_eq!(gen.input_ids_by_name.len(), 1);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);
    }

    #[test]
    fn test_mutate_remove_connection() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true);

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        let in_id = *gen.input_ids_by_name.get(&INPUT_1).unwrap();
        let out_id = *gen.output_ids_by_name.get(&OUTPUT_1).unwrap();

        assert_eq!(gen.connections_by_edge
                   .get(&(in_id, out_id))
                   .unwrap()
                   .enabled,
                   true);

        gen.mutate_remove_connection(&mut rng);

        // Connection is disabled but not removed.
        assert_eq!(gen.connections_by_edge
                   .get(&(in_id, out_id))
                   .unwrap()
                   .enabled,
                   false);
    }

    #[test]
    fn test_mutate_remove_connection_no_connections() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        let in_id = *gen.input_ids_by_name.get(&INPUT_1).unwrap();
        let out_id = *gen.output_ids_by_name.get(&OUTPUT_1).unwrap();

        assert_eq!(gen.connections_by_edge
                   .get(&(in_id, out_id))
                   .is_none(),
                   true);

        gen.mutate_remove_connection(&mut rng);

        // Still doesn't exist.
        assert_eq!(gen.connections_by_edge
                   .get(&(in_id, out_id))
                   .is_none(),
                   true);
    }

    #[test]
    fn test_mutate_remove_connection_no_active_connections() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true);

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        let in_id = *gen.input_ids_by_name.get(&INPUT_1).unwrap();
        let out_id = *gen.output_ids_by_name.get(&OUTPUT_1).unwrap();

        gen.connections_by_edge.get_mut(&(in_id, out_id)).unwrap().enabled = false;

        gen.mutate_remove_connection(&mut rng);

        // Still disabled.
        assert_eq!(gen.connections_by_edge
                   .get(&(in_id, out_id))
                   .unwrap()
                   .enabled,
                   false);
    }

    // TODO(orglofch): Test with recurrences enabled.

    // TODO(orglofch): fn test_mutate_add_connection_reenable() {}

    #[test]
    fn test_mutate_add_node() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true);

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        gen.mutate_add_node(&mut gen_conf, &mut rng);

        assert_eq!(gen.connections_by_edge.len(), 3);
        assert_eq!(gen.input_ids_by_name.len(), 1);
        assert_eq!(gen.hidden_nodes_by_id.len(), 1);
        assert_eq!(gen.output_ids_by_name.len(), 1);
    }

    #[test]
    fn test_mutate_add_node_no_free_connection() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(false);

        let mut rng = rand::thread_rng();

        let mut gen = Genome::new(&mut gen_conf);

        gen.mutate_add_node(&mut gen_conf, &mut rng);

        assert_eq!(gen.connections_by_edge.len(), 0);
        assert_eq!(gen.input_ids_by_name.len(), 1);
        assert_eq!(gen.hidden_nodes_by_id.len(), 0);
        assert_eq!(gen.output_ids_by_name.len(), 1);
    }

    #[test]
    fn test_mutate_weight() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true);

        let mut gen = Genome::new(&mut gen_conf);

        // TODO(orglofch): StepRng doesn't affect gen_range.
        let mut rng = StepRng::new(1, 0);

        let in_id = *gen.input_ids_by_name.get(&INPUT_1).unwrap();
        let out_id = *gen.output_ids_by_name.get(&OUTPUT_1).unwrap();

        let initial_weight = gen.connections_by_edge
            .get(&(in_id, out_id))
            .unwrap()
            .weight;

        assert_eq!(initial_weight, 1.0);

        gen.mutate_weight(&mut rng);

        let new_weight = gen.connections_by_edge
            .get(&(in_id, out_id))
            .unwrap()
            .weight;

        assert_eq!(new_weight, 0.0);
    }

    #[test]
    fn test_creates_cycle() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);
        gen_conf.set_start_connected(true);

        let gen = Genome::new(&mut gen_conf);

        let in_id = *gen.input_ids_by_name.get(&INPUT_1).unwrap();
        let out_id = *gen.output_ids_by_name.get(&OUTPUT_1).unwrap();

        assert_eq!(gen.creates_cycle(out_id, in_id), true);
    }

    #[test]
    fn test_creates_cycle_self_cycle() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT_1], vec![OUTPUT_1]);

        let gen = Genome::new(&mut gen_conf);

        assert_eq!(gen.creates_cycle(0, 0), true);
    }
}
