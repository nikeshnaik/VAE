import tensorflow as tf
import os

class DataGenerator():

    def __init__(self, config):
        tf.compat.v1.set_random_seed(config.seed,)
        self.dataset,self.dev_dataset = self.datagen(config)

    def read_data(self, path=''):

        if path and os.path.exists(path):
            colnames = [*range(1, 785)]
            CParams = {'filenames': path, 'compression_type': 'GZIP', 'header': True}
            features = tf.data.experimental.CsvDataset(**CParams, record_defaults=[tf.int32]*784, select_cols=colnames)
            label = tf.data.experimental.CsvDataset(**CParams, record_defaults=[tf.int32]*1, select_cols=[0])
        else:
            raise FileNotFoundError

        return (features, label)

    def preprocessing(self, features, labels,config):

        def normalize(*args):
            return tf.stack(args) / 255

        def one_hot(*args):
            return tf.one_hot(args,10,axis=1)

        def rerank(*args):
            return tf.reshape(args,[10])

        features = features.map(normalize)
        labels = labels.map(one_hot)
        labels = labels.map(rerank)
        dataset = tf.data.Dataset.zip((features, labels))
        dataset = dataset.shuffle(2000)

        dev_dataset = dataset.take(int(0.3*config.no_examples))
        dataset = dataset.skip(int(0.3*config.no_examples))
        if config.type =='train':
            dataset = dataset.repeat().batch(config.batch_size).prefetch(90)
            dev_dataset = dev_dataset.repeat().batch(config.batch_size).prefetch(90)
        else:
            dataset = dataset.repeat().batch(config.batch_size).prefetch(90)
            dev_dataset = dev_dataset.repeat().batch(config.batch_size).prefetch(90)

        return dataset,dev_dataset

    def datagen(self,config):

        features, label = self.read_data(config.data_path)
        dataset,dev_dataset = self.preprocessing(features, label, config)
        # train_iterator = dataset.make_one_shot_iterator()
        # test_iterator = dev_dataset.make_one_shot_iterator()
        # iterator = dataset.make_initializable_iterator()

        return dataset,dev_dataset


if __name__ == '__main__':

    import time
    from config.readconfig import Config
    start = time.time()
    config = Config('./config/model_config.yaml')
    print("Start Timer")
    d = DataGenerator(config)
    x,y = d.iterator.get_next()
    print(y.shape)
    l1 = tf.layers.dense(inputs=x,units=200, activation=tf.nn.relu)
    l2 = tf.layers.dense(inputs=l1,units=100, activation=tf.nn.relu)
    out = tf.layers.dense(inputs=l2,units=10,activation=tf.nn.softmax)
    print(out.shape)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):

        loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(target=y, output=out,from_logits=True))
        opt = tf.train.AdamOptimizer(0.001).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(out,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        while True:

            cost,acc = sess.run([loss,accuracy])
            print("Cost ->{}".format(cost),end='\n')
            print("acc -->{}".format(acc),end='\n')
            print("counter -->{}".format(counter),end='\n')
