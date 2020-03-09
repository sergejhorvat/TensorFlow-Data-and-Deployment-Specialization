class RPSDataset {
  // when we initialize the class we set it to empty
  constructor() {
    this.labels = []
  }
  // example is the output of the prediction for the image from the truncated model of mobilenet (from 'conv_pw_13_relu')
  // label is [0,1,2] for rock, paper, scisors
  addExample(example, label) {
    // if it's the first example
    if (this.xs == null) {
      // keep - keeps tensor even if tf.tidy() is called
      this.xs = tf.keep(example);
      this.labels.push(label);
    } else {
        // for all  subsequent samples we keep all samples via temp one
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }
  
    // One-hot encoder
    // encode only before training
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
