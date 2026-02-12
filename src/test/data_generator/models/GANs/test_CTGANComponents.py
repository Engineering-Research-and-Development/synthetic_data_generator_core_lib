import pytest
import tensorflow as tf
import numpy as np

from sdg_core_lib.data_generator.models.GANs.CTGANComponents import (
    CTGANGenerator,
    CTGANCritic,
    CTGANModel,
    gumbel_softmax,
)


# CTGANComponents Tests


@pytest.fixture()
def sample_skeleton():
    return [
        {"feature_name": "A", "feature_type": "continuous", "feature_size": 2},
        {"feature_name": "B", "feature_type": "categorical", "feature_size": 3},
        {"feature_name": "C", "feature_type": "continuous", "feature_size": 1},
    ]


@pytest.fixture()
def critic():
    return CTGANCritic(pac_size=5, hidden=128, dropout=0.1)


@pytest.fixture()
def generator(sample_skeleton):
    return CTGANGenerator(
        skeleton=sample_skeleton,
        modes_per_continuous_column=[2, 1],
        categories_per_discrete_column=[3],
        hidden=128
    )


@pytest.fixture()
def ctgan_model(generator, critic):
    return CTGANModel(
        generator=generator,
        critic=critic,
        onehot_discrete_indexes=[3, 4, 5, 6, 7, 8]
    )


def test_critic_call_no_padding(critic):
    batch_size = 10
    feature_dim = 5
    x = tf.random.normal([batch_size, feature_dim])
    
    score = critic(x, training=False)
    
    expected_groups = batch_size // critic.pac_size
    assert score.shape == (expected_groups,)
    assert score.dtype == tf.float64


def test_critic_call_with_padding(critic):
    batch_size = 7  # Not divisible by pac_size=5
    feature_dim = 5
    x = tf.random.normal([batch_size, feature_dim])
    
    score = critic(x, training=False)
    
    expected_groups = (batch_size + critic.pac_size - 1) // critic.pac_size
    assert score.shape == (expected_groups,)
    assert score.dtype == tf.float64


def test_critic_call_training_mode(critic):
    batch_size = 10
    feature_dim = 5
    x = tf.random.normal([batch_size, feature_dim])
    
    score_training = critic(x, training=True)
    score_no_training = critic(x, training=False)
    
    assert score_training.shape == score_no_training.shape
    assert score_training.dtype == tf.float64


def test_gumbel_softmax_basic():
    batch_size = 4
    num_classes = 3
    logits = tf.random.normal([batch_size, num_classes])
    
    result_hard = gumbel_softmax(logits, tau=0.2, hard=True)
    result_soft = gumbel_softmax(logits, tau=0.2, hard=False)
    
    assert result_hard.shape == (batch_size, num_classes)
    assert result_soft.shape == (batch_size, num_classes)
    
    # Check that hard version produces one-hot like outputs
    assert tf.reduce_sum(result_hard, axis=-1).numpy().all() == 1.0
    
    # Check that soft version sums to 1
    assert np.allclose(tf.reduce_sum(result_soft, axis=-1).numpy(), 1.0, atol=1e-6)


def test_gumbel_softmax_different_tau():
    batch_size = 2
    num_classes = 2
    logits = tf.random.normal([batch_size, num_classes])
    
    result_low_tau = gumbel_softmax(logits, tau=0.1, hard=True)
    result_high_tau = gumbel_softmax(logits, tau=1.0, hard=True)
    
    assert result_low_tau.shape == result_high_tau.shape
    assert not np.array_equal(result_low_tau.numpy(), result_high_tau.numpy())


def test_generator_instantiation(generator, sample_skeleton):
    assert generator.skeleton == sample_skeleton
    assert generator.modes_cont == [2, 1]
    assert generator.cats_disc == [3]
    assert generator.tau == 0.2


def test_generator_call(generator):
    batch_size = 4
    noise_dim = 3  # skeleton has 2 continuous + 1 continuous = 3 total continuous features
    cond_dim = 3   # 1 categorical with 3 categories
    
    z = tf.random.normal([batch_size, noise_dim])
    cond = tf.random.normal([batch_size, cond_dim])
    
    full_row, alphas, betas, ds = generator([z, cond], training=False)
    
    # Expected output dimensions: 2 continuous cols * (alpha + beta) + 1 categorical col * categories
    # = 2 * (1 + 2) + 1 * 3 = 6 + 3 = 9
    expected_output_dim = 9
    assert full_row.shape == (batch_size, expected_output_dim)
    assert len(alphas) == 2  # 2 continuous columns
    assert len(betas) == 2   # 2 continuous columns
    assert len(ds) == 1      # 1 categorical column


def test_generator_training_mode(generator):
    batch_size = 2
    noise_dim = 3
    cond_dim = 3
    
    z = tf.random.normal([batch_size, noise_dim])
    cond = tf.random.normal([batch_size, cond_dim])
    
    full_row_train, _, _, _ = generator([z, cond], training=True)
    full_row_eval, _, _, _ = generator([z, cond], training=False)
    
    assert full_row_train.shape == full_row_eval.shape


def test_ctgan_model_instantiation(ctgan_model):
    assert ctgan_model.generator is not None
    assert ctgan_model.critic is not None
    assert ctgan_model.onehot_discrete_indexes == [3, 4, 5, 6, 7, 8]
    assert ctgan_model.gen_loss_tracker is not None
    assert ctgan_model.critic_loss_tracker is not None


def test_ctgan_model_metrics(ctgan_model):
    metrics = ctgan_model.metrics
    assert len(metrics) == 2
    assert metrics[0].name == "generator_loss"
    assert metrics[1].name == "discriminator_loss"


def test_ctgan_model_generate_batch_cond(ctgan_model):
    batch_size = 8
    # Mock PMF list
    ctgan_model.probability_mass_function_list = [
        tf.constant([0.3, 0.7]),  # PMF for first categorical
    ]
    
    cond = ctgan_model.generate_batch_cond(batch_size)
    
    assert cond.shape == (batch_size, 3)  # 3 categories total
    assert tf.reduce_sum(cond, axis=1).numpy().all() == 1.0


def test_ctgan_model_sample_real_data():
    train_tensor = tf.constant([
        [1.0, 0.0, 1.0, 0.0, 0.0],  # Sample with category 0
        [0.0, 1.0, 0.0, 1.0, 0.0],  # Sample with category 1
        [1.0, 0.0, 0.0, 0.0, 1.0],  # Sample with category 2
    ], dtype=tf.float32)
    
    cond = tf.constant([0.0, 1.0, 0.0])  # Select category 1
    discrete_onehot_indexes = [0, 1, 2, 3, 4]
    
    result = CTGANModel.sample_real_data(train_tensor, cond, discrete_onehot_indexes)
    
    assert result.shape == (1, 5)  # One sample, full feature vector


def test_ctgan_model_get_pmfs(ctgan_model):
    # Mock training data with one-hot encoded categorical features
    train_data = tf.constant([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # cat0=0, cat1=1
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # cat0=1, cat1=2
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # cat0=0, cat1=3
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # cat0=1, cat1=1
    ], dtype=tf.float32)
    
    ctgan_model.onehot_discrete_indexes = [0, 1, 2, 3, 4, 5]
    
    pmfs = ctgan_model.get_pmfs(train_data)
    
    assert len(pmfs) == 1  # One categorical column
    assert pmfs[0].shape == (3,)  # 3 categories


def test_ctgan_model_compile(ctgan_model):
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    ctgan_model.compile(g_optimizer, d_optimizer)
    
    assert ctgan_model.generator.optimizer is not None
    assert ctgan_model.critic.optimizer is not None


def test_ctgan_model_generate_without_fitting(ctgan_model):
    with pytest.raises(RuntimeError) as exception_info:
        ctgan_model.generate(batch_size=10)
    assert "In order to generate some data you need to fit a dataset first!" in str(exception_info)


def test_ctgan_model_generate_after_fitting(ctgan_model):
    # Setup mock data
    ctgan_model.probability_mass_function_list = [
        tf.constant([0.3, 0.7]),
    ]
    ctgan_model.row_dim = 6  # Total feature dimension
    
    batch_size = 5
    generated_data = ctgan_model.generate(batch_size)
    
    assert generated_data.shape == (batch_size, 6)
    assert isinstance(generated_data, np.ndarray)


def test_ctgan_model_train_step(ctgan_model):
    # Setup mock training data
    batch_size = 4
    feature_dim = 6
    train_data = tf.random.normal([batch_size, feature_dim])
    ctgan_model._train_data = train_data
    ctgan_model.probability_mass_function_list = [
        tf.constant([0.5, 0.5]),
    ]
    ctgan_model.onehot_discrete_indexes = [2, 3, 4, 5]
    
    # Compile with optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ctgan_model.compile(g_optimizer, d_optimizer)
    
    # Run train step
    metrics = ctgan_model.train_step(train_data)
    
    assert "g_loss" in metrics
    assert "d_loss" in metrics
    assert isinstance(metrics["g_loss"], tf.Tensor)
    assert isinstance(metrics["d_loss"], tf.Tensor)


def test_integration_generator_critic(generator, critic):
    batch_size = 4
    noise_dim = 3
    cond_dim = 3
    
    z = tf.random.normal([batch_size, noise_dim])
    cond = tf.random.normal([batch_size, cond_dim])
    
    # Generate fake data
    fake_data, _, _, _ = generator([z, cond], training=False)
    
    # Pass through critic
    critic_score = critic(fake_data, training=False)
    
    assert fake_data.shape[0] == batch_size
    assert critic_score.dtype == tf.float64
