//! Owned factual adaptation. No game rules, actual world RNG, or personality training.
#![forbid(unsafe_code)]
pub mod evidence;
pub mod experience;
mod owner;
pub mod prediction;
pub mod progress;
pub use owner::*;
pub use reflex_immediate::{Error, Result};

fn bounded(x: f64, lo: f64, hi: f64) -> Result<()> {
    if !x.is_finite() || x < lo || x > hi {
        return Err(Error::Invalid(
            "finite number within declared bounds required",
        ));
    }
    Ok(())
}

fn labels(xs: &[String], limit: usize) -> Result<()> {
    if xs.len() > limit
        || xs.iter().any(|x| x.is_empty() || x.chars().count() > 128)
        || xs.iter().collect::<std::collections::BTreeSet<_>>().len() != xs.len()
    {
        return Err(Error::Invalid("bounded distinct labels required"));
    }
    Ok(())
}
