use std::collections::hash_map::Entry;
use std::collections::HashMap;

// TODO(orglofch): This diverges from the original definition of what
// innovation_numbers represent so it would be worth validating it
// once we have a more working setup.

/// Stores historical origins of innovations which can be used to dedupe
/// the same innovation in future generations.
///
/// A pairing function was also considered to make the innovation ids implicit
/// to what is being connected. However this has some downsides:
/// 1) The pairing function is potentially quite sparse and the growth rate
///    makes it too large to be contained within primitive integer types.
/// 2) There isn't a strong need for reversibility, outside of convenience.
pub(crate) struct InnovationArchive {
    /// The next `NodeGene` innovation id generated between all `Genomes` globally.
    next_node_innovation_id: u32,

    /// The next `ConnectionGene` innovation id generated between all `Genomes` globally.
    next_connection_innovation_id: u32,

    /// The input `NodeGene` ids, preserving order.
    ///
    /// Note, the order itself doesn't define the id since input nodes can be added or
    /// removed after hidden node generation and we don't want there to be collisions.
    input_node_ids: Vec<u32>,

    /// The output `NodeGene` ids, preserving order.
    ///
    /// Note, the order itself doesn't define the id since output nodes can be added or
    /// removed after hidden node generation and we don't want there to be collisions.
    output_node_ids: Vec<u32>,

    /// The hidden `NodeGene` ids mapped by their origin (a pair `(in_id, out_id)`)
    /// representing the connection which the `NodeGene` "split" to be created.
    ///
    /// A `ConnectionGene` can be split multiple times (the original is just disabled)
    /// so we maintain a list of each split and the ordering of already existing
    /// splits in the genomes determines which id we return when recording innovations.
    hidden_node_ids_by_origin: HashMap<(u32, u32), Vec<u32>>,

    /// The `ConnectionGene` ids mapped by their origin (a pair `(in_id, out_id)`) representing
    /// the nodes which the `ConnectionGene` connects.
    ///
    /// `NodeGenes` can only be connected once since the `ConnectionGene` is never deleted,
    /// only disabled.
    connection_ids_by_origin: HashMap<(u32, u32), u32>,
}

// TODO(orglofch): Make this thread safe so we can parallelize mutations.
impl InnovationArchive {
    pub(crate) fn new() -> InnovationArchive {
        InnovationArchive {
            next_node_innovation_id: 0,
            next_connection_innovation_id: 0,
            input_node_ids: Vec::new(),
            output_node_ids: Vec::new(),
            hidden_node_ids_by_origin: HashMap::new(),
            connection_ids_by_origin: HashMap::new(),
        }
    }

    /// Record a new 'spontaneous' input `NodeGene`.
    ///
    /// # Arguments
    ///
    /// * `existing_inputs` - The nunber of existing inputs in the `Genome`.
    ///
    /// Spontaneously generated `NodeGenes` are those which are formed without descent
    /// from existing genes.
    pub(crate) fn record_spontaneous_input_node(&mut self, existing_inputs: u32) -> u32 {
        if self.input_node_ids.len() > existing_inputs as usize {
            return self.input_node_ids[existing_inputs as usize];
        }

        debug_assert_eq!(
            existing_inputs as usize,
            self.input_node_ids.len(),
            "An input id was skipped prior to recording"
        );

        let new_id = self.next_node_innovation_id;
        self.input_node_ids.push(new_id);
        self.next_node_innovation_id += 1;
        new_id
    }

    /// Record a new 'spontaneous' output `NodeGene`.
    ///
    /// # Arguments
    ///
    /// * `existing_outputs` - The number of existing outputs in the `Genome`.
    ///
    /// Spontaneously generated `NodesGenes` are those which are formed without descent
    /// from existing genees.
    pub(crate) fn record_spontaneous_output_node(&mut self, existing_outputs: u32) -> u32 {
        if self.output_node_ids.len() > existing_outputs as usize {
            return self.output_node_ids[existing_outputs as usize];
        }

        debug_assert_eq!(
            existing_outputs as usize,
            self.output_node_ids.len(),
            "An output id was skipped prior to recording"
        );

        let new_id = self.next_node_innovation_id;
        self.output_node_ids.push(new_id);
        self.next_node_innovation_id += 1;
        new_id
    }

    /// Record a hidden `NodeGene` innovation.
    ///
    /// # Arguments
    ///
    /// * `in_id` - The `in_id` of the connection which this `NodeGene` "split".
    ///
    /// * `out_id` - The `out_id` of the connection which this `NodeGene` "split".
    ///
    /// * `split_count` - The number of times the `ConnectionGene` specified by `(in_id, out_id)`
    ///                   has been split prior to this new connection.
    ///
    /// # Returns
    ///
    /// The id of the innovation if it already exists in the archive, or a new id.
    pub(crate) fn record_hidden_node_innovation(
        &mut self,
        in_id: u32,
        out_id: u32,
        split_count: u32,
    ) -> u32 {
        match self.hidden_node_ids_by_origin.entry((in_id, out_id)) {
            Entry::Occupied(o) => {
                let mut value = o.into_mut();
                if split_count < value.len() as u32 {
                    return value[split_count as usize];
                }

                debug_assert_eq!(
                    split_count as usize,
                    value.len(),
                    "A split was skipped prior to recording"
                );

                let new_id = self.next_node_innovation_id;
                self.next_node_innovation_id += 1;
                value.push(new_id);
                new_id
            }
            Entry::Vacant(v) => {
                let mut value = Vec::new();
                let new_id = self.next_node_innovation_id;
                self.next_node_innovation_id += 1;
                value.push(new_id);
                v.insert(value);
                new_id
            }
        }
    }

    /// Record a `ConnectionGene` innovation.
    ///
    /// # Arguments
    ///
    /// * `in_id` - The `in_id` of the new `ConnectionGene`.
    ///
    /// * `out_id` - THe `out_id` of the new `ConnectionGene`.
    ///
    /// # Returns
    ///
    /// The id of the innovation if it already exists in the archive, or a new id.
    pub(crate) fn record_connection_innovation(&mut self, in_id: u32, out_id: u32) -> u32 {
        match self.connection_ids_by_origin.get(&(in_id, out_id)) {
            Some(&id) => id,
            None => {
                let new_id = self.next_connection_innovation_id;
                self.connection_ids_by_origin.insert(
                    (in_id, out_id),
                    new_id,
                );
                self.next_connection_innovation_id += 1;
                new_id
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_record_spontaneous_input_node() {
        let mut archive = InnovationArchive::new();

        assert_eq!(archive.record_spontaneous_input_node(0), 0);

        // Same node and order creates same innovation.
        assert_eq!(archive.record_spontaneous_input_node(0), 0);

        // An entirely new order creates a new innovation.
        assert_eq!(archive.record_spontaneous_input_node(1), 1);

        // It won't use the same id as an output node.
        archive.record_spontaneous_output_node(0);
        assert_eq!(archive.record_spontaneous_input_node(2), 3);

        // It won't use the same id as a hidden node.
        archive.record_hidden_node_innovation(1, 1, 0);
        assert_eq!(archive.record_spontaneous_input_node(3), 5);

        // Connection ids are untouched.
        assert_eq!(archive.record_connection_innovation(1, 1), 0);
    }

    #[test]
    pub fn test_record_spontaneous_output_node() {
        let mut archive = InnovationArchive::new();

        assert_eq!(archive.record_spontaneous_output_node(0), 0);

        // Same node and order creates same innovation.
        assert_eq!(archive.record_spontaneous_output_node(0), 0);

        // An entirely new order creates a new innovation.
        assert_eq!(archive.record_spontaneous_output_node(1), 1);

        // It won't use the same id as an input node.
        archive.record_spontaneous_input_node(0);
        assert_eq!(archive.record_spontaneous_output_node(2), 3);

        // It won't use the same id as a hidden node.
        archive.record_hidden_node_innovation(1, 1, 0);
        assert_eq!(archive.record_spontaneous_output_node(3), 5);

        // Connection ids are untouched.
        assert_eq!(archive.record_connection_innovation(1, 1), 0);
    }

    #[test]
    pub fn test_record_hidden_node_innovation() {
        let mut archive = InnovationArchive::new();

        assert_eq!(archive.record_hidden_node_innovation(1, 2, 0), 0);

        // Same node and split creates same innovation.
        assert_eq!(archive.record_hidden_node_innovation(1, 2, 0), 0);

        // Same node but different split creates new innovation.
        assert_eq!(archive.record_hidden_node_innovation(1, 2, 1), 1);

        // Reverse of an existing node creates a new innovation.
        assert_eq!(archive.record_hidden_node_innovation(2, 1, 0), 2);

        // An entirely new node creates a new innovation.
        assert_eq!(archive.record_hidden_node_innovation(3, 4, 0), 3);

        // Connection ids are untouched.
        assert_eq!(archive.record_connection_innovation(1, 1), 0);
    }

    #[test]
    pub fn test_record_connection() {
        let mut archive = InnovationArchive::new();

        assert_eq!(archive.record_connection_innovation(1, 2), 0);

        // Same connection creates the same innovation.
        assert_eq!(archive.record_connection_innovation(1, 2), 0);

        // Reverse of an existing connection creates a new innovation.
        assert_eq!(archive.record_connection_innovation(2, 1), 1);

        // An entirely new connection creates a new innovation.
        assert_eq!(archive.record_connection_innovation(3, 4), 2);

        // Node ids are untouched.
        assert_eq!(archive.record_hidden_node_innovation(1, 1, 0), 0);
    }
}
