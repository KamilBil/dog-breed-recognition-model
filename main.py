import tensorflow_datasets as tfds
import tensorflow as tf

SIZE = (299, 299)  # for Xception
BATCH_SIZE = 32
DROPOUT_RATE = 0.5
EPOCHS = 1

# Load data
dataset, metadata = tfds.load('stanford_dogs', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    # Random rotations (maximum 20 degrees)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_crop(image, size=[int(0.8 * SIZE[0]), int(0.8 * SIZE[1]), 3])
    image = tf.image.resize(image, SIZE)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalization [0,1]
    image = tf.image.resize(image, SIZE)
    return image, label


def create_augmented_datasets(dataset, num_copies=1):
    augmented_datasets = [dataset.map(preprocess).map(augment) for _ in range(num_copies)]
    combined_dataset = augmented_datasets[0]
    for aug_dataset in augmented_datasets[1:]:
        combined_dataset = combined_dataset.concatenate(aug_dataset)
    return combined_dataset


train_dataset_augmented = create_augmented_datasets(train_dataset)
train_dataset_original = train_dataset.map(preprocess)
train_dataset = train_dataset_original.concatenate(train_dataset_augmented).shuffle(10000).batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE)

input_tensor = tf.keras.layers.Input(shape=(SIZE[0], SIZE[1], 3))
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(rate=DROPOUT_RATE),
    tf.keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)

model.save('dog_breeds_v1', save_format='tf')

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy}')

loaded_model = tf.keras.models.load_model('dog_breeds_v1')
