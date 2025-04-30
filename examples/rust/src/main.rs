use anyhow::Result;
use std::time::Instant;
use tch::nn::{Adam, ModuleT, OptimizerConfig, VarStore};
use tch::vision::dataset::Dataset;
use tch::{Device, TrainableCModule};

fn train_and_save_model(dataset: &Dataset, device: Device) -> Result<()> {
    let vs = VarStore::new(device);

    let mut trainable = TrainableCModule::load("model.pt", vs.root())?;
    trainable.set_train();
    let initial_acc = trainable.batch_accuracy_for_logits(
        &dataset.test_images,
        &dataset.test_labels,
        vs.device(),
        1024,
    );
    println!("Initial accuracy: {:5.2}%", 100. * initial_acc);

    let mut opt = Adam::default().build(&vs, 1e-3)?;

    for epoch in 0..2 {
        let start = Instant::now();
        let mut total_train_loss = 0f64;
        let mut batch_count = 0usize;
        let mut last_batch_size = 0usize;

        // === Training ===
        for (images, labels) in dataset
            .train_iter(2048)
            .shuffle()
            .to_device(vs.device())
        {
            last_batch_size = images.size()[0] as usize;
            let loss = trainable
                .forward_t(&images, true)
                .cross_entropy_for_logits(&labels);
            total_train_loss += loss.double_value(&[]);
            opt.backward_step(&loss);
            batch_count += 1;
        }

        // === Validation (no gradient) ===
        trainable.set_eval();
        let val_logits = trainable.forward_t(&dataset.test_images, false);
        let val_loss = val_logits.cross_entropy_for_logits(&dataset.test_labels).double_value(&[]);
        let val_accuracy = trainable.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            vs.device(),
            1024,
        );
        trainable.set_train();

        let duration = start.elapsed();
        println!(
            "epoch: {:2} | val acc: {:5.2}% | train loss: {:6.4} | val loss: {:6.4} | batches: {:3} | batch size: {:3} | time: {:.2?}",
            epoch,
            100. * val_accuracy,
            total_train_loss / batch_count as f64,
            val_loss,
            batch_count,
            last_batch_size,
            duration
        );
    }

    trainable.save("trained_model.pt")?;
    Ok(())
}

fn main() -> Result<()> {
    let dataset = tch::vision::mnist::load_dir("data/FashionMNIST/raw")?;
    let dataset = Dataset {
        train_images: dataset.train_images.view([-1, 1, 28, 28]),
        test_images: dataset.test_images.view([-1, 1, 28, 28]),
        ..dataset
    };
    let device = Device::Cpu;
    train_and_save_model(&dataset, device)
}
