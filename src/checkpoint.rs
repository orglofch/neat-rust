use config::Config;
use genome::Population;

pub struct CheckpointConfig {
    /// The function to invoke on every checkpoint.
    ///
    /// Checkpointing functions are free to change the configuration based on the
    /// state of the configuration or however else they can think of.
    ///
    /// TODO(orglofch): Add example.
    checkpoint_fn: Option<fn(&mut Config, &Population)>,

    /// The number of generations between checkpointing function invocations.
    checkpoint_rate: u32,
}

impl CheckpointConfig {
    pub fn new() -> CheckpointConfig {
        CheckpointConfig {
            checkpoint_fn: None,
            checkpoint_rate: 0,
        }
    }

    pub fn set_checkpoint_fn(&mut self, f: fn(&mut Config, &Population)) -> &mut CheckpointConfig {
        self.checkpoint_fn = Some(f);
        self
    }

    pub fn set_checkpoint_rate(&mut self, checkpoint_rate: u32) -> &mut CheckpointConfig {
        self.checkpoint_rate = checkpoint_rate;
        self
    }
}
