#include <iostream>
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void construct_net(network<cross_entropy_multiclass, adam>& nn){
    // https://github.com/toshi-k/kaggle-digit-recognizer/blob/master/2_model.lua
    nn << convolutional_layer<relu>(28, 28, 5, 1, 64)   // 1x28x28 -> 64x24x24
       << max_pooling_layer<identity>(24, 24, 64, 2)    // -> 64x12x12
       << dropout_layer(12*12*64, 0.1)                  // -> 64x12x12

       << convolutional_layer<relu>(12, 12, 5, 64, 64, padding::same)  // -> 64x12x12
       << max_pooling_layer<identity>(12, 12, 64, 2)    // -> 64x6x6
       << dropout_layer(6*6*64, 0.2)                    // -> 64x6x6

       << convolutional_layer<relu>(6, 6, 3, 64, 256, padding::same)  // -> 256x6x6
       << max_pooling_layer<identity>(6, 6, 256, 2)     // -> 256x3x3
       << dropout_layer(3*3*256, 0.3)                   // -> 256x3x3

       << convolutional_layer<relu>(3, 3, 3, 256, 1024) // -> 1024x1x1
       << dropout_layer(1024, 0.4)                      // -> 1024x1x1

       << fully_connected_layer<relu>(1024, 256)
       << dropout_layer(256, 0.5)

       << fully_connected_layer<softmax>(256, 10);

    // https://github.com/zaburo-ch/FaxOCR/blob/master/code/faxmnist_keras.py
    // nn << convolutional_layer<relu>(28, 28, 5, 1, 64)   // 1x28x28 -> 64x24x24
    //    << max_pooling_layer<identity>(24, 24, 64, 2)    // -> 64x12x12
    //    << dropout_layer(12*12*64, 0.2)                  // -> 64x12x12

    //    << convolutional_layer<relu>(12, 12, 5, 64, 64, padding::same)  // -> 64x12x12
    //    << dropout_layer(12*12*64, 0.3)                    // -> 64x12x12

    //    << convolutional_layer<relu>(12, 12, 3, 64, 256)   // -> 256x10x10
    //    << dropout_layer(10*10*256, 0.4)                   // -> 256x10x10

    //    << fully_connected_layer<relu>(10*10*256, 128)
    //    << dropout_layer(128, 0.5)

    //    << fully_connected_layer<softmax>(128, 10);
}

void train_lenet(std::string data_dir_path) {
    network<cross_entropy_multiclass, adam> nn;
    construct_net(nn);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    // parse_mnist_labels(data_dir_path+"/faxocr-training-28_train_labels.idx1",
    //                    &train_labels);
    // parse_mnist_images(data_dir_path+"/faxocr-training-28_train_images.idx3",
    //                    &train_images, -1.0, 1.0, 2, 2);
    // parse_mnist_labels(data_dir_path+"/faxocr-mustread-28_train_labels.idx1",
    //                    &test_labels);
    // parse_mnist_images(data_dir_path+"/faxocr-mustread-28_train_images.idx3",
    //                    &test_images, -1.0, 1.0, 2, 2);

    parse_mnist_labels(data_dir_path+"/train-labels.idx1-ubyte",
                       &train_labels);
    parse_mnist_images(data_dir_path+"/train-images.idx3-ubyte",
                       &train_images, 0.0, 1.0, 0, 0);
    parse_mnist_labels(data_dir_path+"/t10k-labels.idx1-ubyte",
                       &test_labels);
    parse_mnist_images(data_dir_path+"/t10k-images.idx3-ubyte",
                       &test_images, 0.0, 1.0, 0, 0);

    std::cout << "start training" << std::endl;

    progress_display disp(train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << " " << 1.0*res.num_success/res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train(train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data)" << std::endl;
        return -1;
    }
    train_lenet(argv[1]);
}
