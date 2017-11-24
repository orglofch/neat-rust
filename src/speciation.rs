// TODO(orglofch): Allow arbitrary speciation config.

pub struct SpeciationConfig {
    /// The maximum compatible distance between two genomes which are still part of the same species.
    compatibility_threshold: f32,

    // TODO(orglofch): Maybe add compatibility_excess_coefficient: f32.

    /// The coefficient applied to the number of disjoint genes when calculating compatibility.
    compatibility_disjoint_coefficient: f32,

    /// The coefficient applied to the weight difference of matching genes when calculating compatibility.
    compatibility_weight_coefficient: f32,
}

// TODO(orglofch): Create appropriate defaults.
impl SpeciationConfig {
    pub fn new() -> SpeciationConfig {
        SpeciationConfig {
            compatibility_threshold: 0.0,
            compatibility_disjoint_coefficient: 0.0,
            compatibility_weight_coefficient: 0.0,
        }
    }

    pub fn set_compatibility_threshold(&mut self, threshold: f32) -> &mut SpeciationConfig {
        self.compatibility_threshold = threshold;
        self
    }

    pub fn set_compatibility_disjoint_coefficient(
        &mut self,
        coefficient: f32,
    ) -> &mut SpeciationConfig {
        self.compatibility_disjoint_coefficient = coefficient;
        self
    }

    pub fn set_compatibility_weight_coefficient(
        &mut self,
        coefficient: f32,
    ) -> &mut SpeciationConfig {
        self.compatibility_weight_coefficient - coefficient;
        self
    }
}
