use std::sync::{Arc, Mutex};

use clap::Parser;
use dinner_party::DinnerParty;
use rand::SeedableRng;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of people participating
    #[arg(short, long)]
    particpants: usize,

    /// Size of the groups being made
    #[arg(short, long)]
    group_size: usize,

    /// Number of rounds people participate in
    #[arg(short, long)]
    rounds: usize,

    /// Maximum number of steps for simulated annealing
    #[arg(short, long, default_value_t = 1000000)]
    steps: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let problem = DinnerParty {
        participants: args.particpants,
        group_size: args.group_size,
        rounds: args.rounds,
        rng: Arc::new(Mutex::new(SeedableRng::from_os_rng())),
    };

    println!(
        "Running approximation for optimal Dinner Party of {} guests, tables of size {}, with {} rounds. Number of runs={}.",
        &problem.participants, &problem.group_size, &problem.rounds, args.steps
    );

    let seating_chart = dinner_party::run(&problem, args.steps)?;

    println!(
        "Everyone meets at least {} people with the following assignments",
        problem.min_people_met(&problem.people_met(&seating_chart))
    );

    let mut header = "Person,".to_string();
    header.push_str(
        &(1..=problem.rounds)
            .map(|n| format!("Round {n}"))
            .collect::<Vec<String>>()
            .join(","),
    );

    println!("{header}");

    let mut wtr = csv::Writer::from_writer(std::io::stdout());

    let mut ordered_output: Vec<(usize, Vec<char>)> =
        problem.to_table_order(&seating_chart).drain().collect();
    ordered_output.sort();

    for row in ordered_output.clone() {
        let person: usize = row.0;
        let tables: Vec<String> = row.1.iter().map(ToString::to_string).collect();
        let row = [&[person.to_string()], &tables[..]].concat();
        wtr.write_record(row)?;
    }

    Ok(())
}
