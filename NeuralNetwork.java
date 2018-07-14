import java.util.Random;
import java.util.ArrayList;

/**
 * <h1>NeuralNetwork</h1>
 * 
 * <p>This class greatly improves the ease of creating, training, and testing
 * simple neural networks. Simply input an array of the network's structure
 * to create a new neural network. You can then use the train() method to 
 * train the network, and test() and predict() methods to test data.</p>
 *
 * <p>
 * Notes:</p>
 * <ul>
 * <li>Structure array should be of the form: {# nodes layer 1, # nodes layer 2, ...}. For example, a  4 --&gt; 4 --&gt; 1  neural network would have structure array {4, 4, 1}</li>
 * 
 * <li>NeuralNetwork is double[] input to double[] output (manipulate your data to be 1 dimension)</li>
 * 
 * <li>The train(), test(), and predict() methods take double[][] arguments which
 * are arrays of double[] input and output data</li>
 * 
 * <li>Use checkData() on input and output data before using NeuralNetwork to ensure
 * the data fits the network structure and is of the right length</li>
 * 
 * <li>If training with data batches, stochastic gradient descent is used</li>
 * </ul>
 * 
 * @author Kyle Sadler
 * @version 1.0
 * @since 2018-07-13
 */

public class NeuralNetwork{
    
    /** 
     * Notes for code modification:
     * 
     * activation function: sigmoid function 
     * -- this can be changed by modifying activationFunction() method
     * -- must also modify activationDerivative() method to be derivative of activation function
     * 
     * error function: 0.5*(expected-actual)^2
     * -- this can be changed by modifying errorFunction() method
     * -- must also modify errorDerivative() method to be derivative of error function
     * 
     * structure of network: determined by the structure array
     * -- contains number of nodes for each level
     * -- ex for a  4 -> 4 -> 1  network, int[] structure = {4, 4, 1}
     * 
     * Data ArrayLists: contain data for each layer
     * -- created from structure array
     * -- layer 0 is first (input) layer of the network
     * 
     * layerActivations: contain activations for each layer
     * -- layerActivations.get(0) is the activations for layer 0,
     * -- layerActivations.get(1) is the activations for layer 1,
     * -- etc.
     * 
     * layerInputs: contain inputs for each layer
     * -- layerInputs.get(1) is the inputs for layer 1,
     * -- layerInputs.get(2) is the inputs for layer 2,
     * -- etc.
     * -- layerInputs.get(0) is null
     * 
     * weights: contain weights between each layer and the next (2d matrix)
     * -- weights.get(0) is the weights between layer 0 and layer 1
     * -- weights.get(1) is the weights between layer 1 and layer 2
     * -- etc.
     * -- weights.get(structure.length-1) does not exist
     * 
     * weightDeltas: adjustment to weights between each layer and the next (2d matrix)
     * -- weightDeltas.get(0) is the weightDeltas between layer 0 and layer 1
     * -- weightDeltas.get(1) is the weightDeltas between layer 1 and layer 2
     * -- etc.
     * -- weightDeltas.get(structure.length-1) does not exist
     * 
     * biases: contain biases for each layer
     * -- biases.get(1) is the biases for layer 1
     * -- biases.get(2) is the biases for layer 2
     * -- etc.
     * -- biases.get(0) is null
     * 
     * biasDeltas: adjustment to biases for each layer
     * -- biasDeltas.get(1) is the biasDeltas for layer 1
     * -- biasDeltas.get(2) is the biasDeltas for layer 2
     * -- etc.
     * -- biasDeltas.get(0) is null
     * 
     * This class is dependent on the Java class Matrix
     */
    
    private static final int maxCharsPerOutput = 300;
    private Random rand;
    private double learningRate;
    private double biasLearningRate;//essentially learningRate. left in for future expansion
    private int[] structure; 
    private ArrayList<double[][]> layerActivations;
    private ArrayList<double[][]> layerInputs;
    private ArrayList<double[][]> weights;
    private ArrayList<double[][]> weightDeltas;
    private ArrayList<double[][]> biases;
    private ArrayList<double[][]> biasDeltas;
    
    
    //constructors
    /**
     * Creates NeuralNetwork with structure array s and learning rate of 1.
     * 
     * @param s structure of NeuralNetwork represented as array of the form {# nodes layer 1, # nodes layer 2, ...}
     */ 
    public NeuralNetwork(int[] s){
        rand = new Random();
        learningRate = 1;
        biasLearningRate = 1;
        
        //initialize everything
        initialize(s, rand);
    }
    /**
     * Creates NeuralNetwork with structure array s and learning rate lr.
     * 
     * @param s structure of NeuralNetwork represented as array of the form {# nodes layer 1, # nodes layer 2, ...}
     * 
     * @param lr network learning rate
     */ 
    public NeuralNetwork(int[] s, double lr){
        rand = new Random();
        learningRate = lr;
        biasLearningRate = lr;
        
        //initialize everything
        initialize(s, rand);
    }
    /**
     * Creates NeuralNetwork with structure array s, learning rate lr, and Random seed.
     * 
     * @param s structure of NeuralNetwork represented as array of the form {# nodes layer 1, # nodes layer 2, ...}
     * 
     * @param lr network learning rate
     * 
     * @param seed seed for random starting weights and biases of NeuralNetwork 
     */ 
    public NeuralNetwork(int[] s, double lr, long seed){
        rand = new Random(seed);
        learningRate = lr;
        biasLearningRate = lr;
        
        //initialize everything
        initialize(s, rand);
    }
    
    
    //public functions
    /**
     * Trains the network with trainingInput and trainingOutput data for a specified number of iterations.
     * Assumes user has run checkData() on trainingInput and trainingOutput.
     * 
     * @param trainingInput array of double[] training input data
     * @param trainingOutput array of double[] training output data
     * @param iterations number of times to train the network with the entire training dataset
     */
    public void train(double[][] trainingInput, double[][] trainingOutput, int iterations){
        if(iterations<1){
            throw new IllegalArgumentException("cannot train for " + iterations + " iterations");    
        }
        
        double[][][] trainingIn = convertData(trainingInput);
        double[][][] trainingOut = convertData(trainingOutput);
        
        System.out.println("Training for " + iterations + " iterations...\n");
        
        
        //run through training set iterations times
        //takes one "step" per iteration
        for(int i=0; i<iterations; i++){
            
            double errorTotal=0;
            
            //run through training set
            for(int j=0; j<trainingIn.length; j++){
                //forward propagation
                propagateForward(trainingIn[j]);
                
                //backward propagation
                propagateBackward(trainingOut[j]);
                
                if(i%(iterations/10)==0){
                    errorTotal+=calcError(layerActivations.get(layerActivations.size()-1), trainingOut[j]);
                    checkError(errorTotal);
                }
            }
        
            updateWeights(trainingIn.length);
            updatebiases(trainingIn.length);
            //learningRate *=.999999;
            //biasLearningRate *=.999999;
            
            if(i%(iterations/10)==0){
                System.out.println("iteration: " + i); 
                System.out.println("error: " + errorTotal + "\n");
            }
        }
        
        System.out.println("\n\n\n\n\n\n");
    }
    /**
     * Trains the network with trainingInput and trainingOutput training data for a number of iterations using specified number of data batches and a method specified by normalizedGradientDescent. Assumes user has run checkData() on trainingInput and trainingOutput.
     * 
     * @param trainingInput array of double[] training input data
     * @param trainingOutput array of double[] training output data
     * @param iterations number of times to train the network with the entire training dataset
     * @param dataBatches number of batches to split the data into. One training step is completed 
     * for each batch (to optimize network efficiently for large training datasets)
     * @param normalizedGradientDescent if true, the gradient descent steps are of uniform length,
     * otherwise they are proportional to the magnitude of the gradient
     */
    public void train(double[][] trainingInput, double[][] trainingOutput, int iterations, int dataBatches, boolean normalizedGradientDescent){
        
        double[][][] trainingIn = convertData(trainingInput);
        double[][][] trainingOut = convertData(trainingOutput);
        
        if(iterations<1){
            throw new IllegalArgumentException("cannot train for " + iterations + " iterations");    
        }
        
        System.out.println("Training for " + iterations + " iterations...\n");
        
        
        //run through training set iterations time
        for(int i=0; i<iterations; i++){
            
            double errorTotal=0;
            //run through training set batches
            //takes one "step" each batch
            for(int batch=0; batch<dataBatches; batch++){
                
                //run through each batch
                for(int j=batch*trainingIn.length/dataBatches; j<(batch+1)*trainingIn.length/dataBatches; j++){
                    //forward propagation
                    propagateForward(trainingIn[j]);
                    
                    //backward propagation
                    propagateBackward(trainingOut[j]);
                    
                    if(i%(iterations/10)==0){
                        errorTotal+=calcError(layerActivations.get(layerActivations.size()-1), trainingOut[j]);
                        checkError(errorTotal);
                    }
                    
                }
                
                //splits remaining training inputs between batches
                if(batch < trainingIn.length%dataBatches){
                    //propagation for the remainder
                    propagateForward(trainingIn[trainingIn.length-1-batch]);
                    propagateBackward(trainingOut[trainingOut.length-1-batch]);
                    
                    if(i%(iterations/10)==0){
                        errorTotal+=calcError(layerActivations.get(layerActivations.size()-1), trainingOut[trainingIn.length-1-batch]);
                        checkError(errorTotal);
                    }
                    
                    //to account for extra training input in batch
                    updateWeights(trainingIn.length/dataBatches+1, normalizedGradientDescent);
                    updatebiases(trainingIn.length/dataBatches+1, normalizedGradientDescent);
                }else{
                    updateWeights(trainingIn.length/dataBatches, normalizedGradientDescent);
                    updatebiases(trainingIn.length/dataBatches, normalizedGradientDescent);
                }
            }
            
            if(i%(iterations/10)==0){
                System.out.println("iteration: " + i); 
                System.out.println("average error: " + errorTotal/trainingIn.length + "\n");
            }
        }
        
        System.out.println("\n\n\n\n\n\n");
    }
    
    /**
     * Tests the network with given testingInput and testingOutput and prints minimal report to console. Assumes user has run checkData() on testingInput and testingOutput.
     * 
     * @param testingInput array of double[] inputs to test
     * @param testingOutput array of double[] outputs to test
     */
    public void test(double[][] testingInput, double[][] testingOutput){
        
        double[][][] testingIn = convertData(testingInput);
        double[][][] testingOut = convertData(testingOutput);
        
        System.out.println("Testing... \n");
        
        double totalError = 0;
        
        for(int i=0; i<testingIn.length; i++){
            propagateForward(testingIn[i]);
            double inputError = calcError(testingOut[i], layerActivations.get(layerActivations.size()-1)); //calc error
            totalError += inputError; // add error to totalError
            
            checkError(totalError);
            checkError(inputError);
            
            System.out.println("test "+i+"..............");// output stats
            System.out.println("error: "+ inputError);
            System.out.println();
        }
        
        System.out.println("average error: " + totalError/testingIn.length+"\n\n");
    }
    /**
     * Tests the network with given testingInput and testingOutput and prints report to console according to showStats.
     * Assumes user has run checkData() on testingInput and testingOutput.
     * 
     * @param testingInput array of double[] inputs to test
     * @param testingOutput array of double[] outputs to test
     * @param showStats if true, show all stats. If false, print minimal report
     * @param label name of testing data in printed report
     * @param errorTolerance error threshold to classify network output as "correct" for a given testing example
     * 
     */
    public void test(double[][] testingInput, double[][] testingOutput, boolean showStats, String label, double errorTolerance){
        
        double[][][] testingIn = convertData(testingInput);
        double[][][] testingOut = convertData(testingOutput);
        
        System.out.println("Testing "+ label + " with " + errorTolerance + " error tolerance... \n");
        
        double totalError = 0;
        int testsPassed = 0;
        
        if(showStats){
            for(int i=0; i<testingIn.length; i++){
                propagateForward(testingIn[i]);
                double inputError = calcError(testingOut[i], layerActivations.get(layerActivations.size()-1)); //calc error
                totalError += inputError; // add error to totalError
                
                checkError(totalError);
                checkError(inputError);
                
                System.out.println("test "+i+"..............");
                System.out.println("input: "+ displayConsole(testingIn[i]));
                System.out.println("expected output "+ displayConsole(testingOut[i]));
                System.out.println("actual output: "+ displayConsole(layerActivations.get(layerActivations.size()-1)));
                System.out.println("error: "+ inputError);
                System.out.print("status: ");
                
                if(inputError < errorTolerance){
                    testsPassed++;
                    System.out.println("passed");
                }else{
                    System.out.println("failed");
                }
                
                System.out.println();
            }
        }else{
            for(int i=0; i<testingIn.length; i++){
                propagateForward(testingIn[i]);
                double inputError = calcError(testingOut[i], layerActivations.get(layerActivations.size()-1)); //calc error
                totalError += inputError; // add error to totalError
                
                checkError(totalError);
                checkError(inputError);
                
                if(inputError < errorTolerance){
                    testsPassed++;
                }
                
                System.out.println("test "+i+"..............");
                System.out.println("error: "+ inputError);
                System.out.println();
            }
        }
        
        System.out.println("average error: " + totalError/testingIn.length);
        System.out.println("tests passed: " + (1000000.0*testsPassed/testingIn.length)/10000 +"%\n\n");
    }
    
    /**
     * Prints network output predictions for predictingInput to console.
     * @param predictingInput input to use for NeuralNetwork predictions
     */ 
    public void predict(double[][] predictingInput){
        double[][][] predictingIn = convertData(predictingInput);
        
        System.out.println("predicting... \n");
        for(int i=0; i<predictingIn.length; i++){
            propagateForward(predictingIn[i]);
            
            System.out.println("prediction "+i+"..............");
            if(predictingIn[i].length*predictingIn[i][0].length < maxCharsPerOutput){
                System.out.println("input: "+ Matrix.toStringOneLine(predictingIn[i]));
            }else{
                System.out.println("input: too long to display");
            }
            System.out.println("predicted output: "+ Matrix.toStringOneLine(layerActivations.get(layerActivations.size()-1)));
            System.out.println();
        }
        System.out.println();

    }
    /**
     * Prints network output predictions for predictingInput to console using specified data label.
     * @param predictingInput input to use for NeuralNetwork predictions
     * @param label name of data
     */ 
    public void predict(double[][] predictingInput, String label){
        double[][][] predictingIn = convertData(predictingInput);
        
        System.out.println("predicting "+label+"... \n");
        for(int i=0; i<predictingIn.length; i++){
            propagateForward(predictingIn[i]);
            
            System.out.println("prediction "+i+"..............");
            if(predictingIn[i].length*predictingIn[i][0].length < maxCharsPerOutput){
                System.out.println("input: "+ Matrix.toStringOneLine(predictingIn[i]));
            }else{
                System.out.println("input: too long to display");
            }
            System.out.println("predicted output: "+ Matrix.toStringOneLine(layerActivations.get(layerActivations.size()-1)));
            System.out.println();
        }
        System.out.println();
    }
    
    /**
     * Returns learning rate of the network.
     * @return learning rate of network
     */ 
    public double getLearningRate(){
        return learningRate;
    }
    /**
     * Sets the network learning rate to lr.
     * @param lr double to set learningRate to
     */ 
    public void setLearningRate(double lr){
        if(lr > 0){
            learningRate = lr;
            biasLearningRate = lr;
        }else{
            throw new IllegalArgumentException("learning rate must be positive: "+lr);
        }
    }
    /**
     * Returns Random object used by the network.
     * @return rand 
     */ 
    public Random getRandom(){
        return rand;
    }
    /**
     * sets Random object used by the network to r.
     * @param r new network Random object
     */
    public void setRandom(Random r){
        if(Random.class.isInstance(r)){
            rand = r;
        }else{
            throw new IllegalArgumentException("argument is not a Random object");
        }
    }
    /**
     * Returns structure array for the network.
     * @return structure array for network of the form {# nodes layer 1, # nodes layer 2, ...}
     */ 
    public int[] getStructure(){
        return structure;
    }
    /**
     * Ensures that input and output data fit network structure
     * and have same length, otherwise throws IllegalArgumentException.
     * @param input input data to be checked
     * @param output output data to be checked
     * @throws IllegalArgumentException if data does not fit network structure or has different lengths
     * */
    public void checkData(double[][] input, double[][] output){
        
        if(input.length != output.length){
            throw new IllegalArgumentException("unequal amount of input and output data: " + input.length + ", " + output.length);
        }
        
        for(int i=0; i<input.length; i++){
            if(input[i].length != structure[0]){
                throw new IllegalArgumentException("Input data has incorrect dimensions at index: " + i);
            } 
            
            if(output[i].length != structure[structure.length-1]){
                throw new IllegalArgumentException("Output data has incorrect dimensions at index: " + i);
            } 
        }
        
    }
    /**
     * Returns data in a shuffled manner
     * @param data data to be shuffled
     * @return shuffled data
     */
    public double[][] shuffleData(double[][] data){
        Random r = new Random();
        
        double[] temp;
        
        for(int i=0; i<data.length; i++){
            //choose new random index
            int index = r.nextInt(data.length);
            
            //swap matrices
            temp = data[i];
            data[i] = data[index];
            data[index] = temp;
        }
        
        return data;
    }
   
   
    //private functions
    /**
     * initializes everything except for rand and learningRate
     */ 
    private void initialize(int[] s, Random rand){
        //initialize
        structure=s;
        layerActivations = new ArrayList<double[][]>(s.length);
        layerInputs = new ArrayList<double[][]>(s.length);
        weights = new ArrayList<double[][]>(s.length-1);
        weightDeltas = new ArrayList<double[][]>(s.length-1);
        biases = new ArrayList<double[][]>(s.length);
        biasDeltas = new ArrayList<double[][]>(s.length);
        
        
        //set up structure
        for(int layer=0; layer<s.length; layer++){
            layerActivations.add(Matrix.randomizeArray(new double[s[layer]][1], rand));
            
            if(layer != 0){
                layerInputs.add(Matrix.randomizeArray(new double[s[layer]][1], rand));
                weights.add(Matrix.randomizeArray(new double[s[layer]][s[layer-1]], rand));
                weightDeltas.add(new double[s[layer]][s[layer-1]]);
                biases.add(Matrix.randomizeArray(new double[s[layer]][1], rand));
                biasDeltas.add(new double[s[layer]][1]);
            }else{
                biases.add(null); // so layer[k] and biases[k] line up
                biasDeltas.add(null);
                layerInputs.add(null);
            }
        }
    }
    /**
     * calculate all resulting activations given an input. store calculations in appropriate ArrayLists
     */
    private void propagateForward(double[][] input){
        
        layerActivations.set(0, input);
        
        for(int layer=1; layer<structure.length; layer++){
            layerInputs.set(layer, Matrix.add(Matrix.multiply(weights.get(layer-1), layerActivations.get(layer-1)), biases.get(layer)));
            
            for(int row=0; row<layerActivations.get(layer).length; row++){
                for(int col=0; col<layerActivations.get(layer)[0].length; col++){
                    layerActivations.get(layer)[row][col] = activationFunction(layerInputs.get(layer)[row][col]);
                }
            }
        }
        
    } 
    /**
     * COMPUTES UNNORMALIZED GRADIENT FOR ONE TRAINING EXAMPLE 
     * SUBTRACTS UNNORMALIZED GRADIENT FROM WEIGHTDELTAS AND BIASDELTAS 
     * (same as adding negative gradient)
     * 
     * @param output -- expected output of network for current training example 
     * 
     * backpropagates using current activations for given output adds 
     * negative gradient for given output to weightDeltas and biasDeltas 
     * (which will then be averaged across the whole batch or data set,
     * possibly normalized, and then multiplied by learningRate)
     * 
     * algorithm:
     * -- start with activation gradient of current layer, L
     * -- find negative gradient of biases[L] and add to biasDeltas
     * -- find negative gradient of weights[L-1] and add to weightDeltas
     * -- find activation gradient of previous layer, L - 1
     * -- repeat
     */ 
    private void propagateBackward(double[][] output){
        /* private double errorDerivative(double expected, double actual){return actual - expected;}
        -- weights.get(structure.length-1) does not exist
        -- biases.get(0) is null
        */
        
        //gradient for layer activations
        double[][] layerGradient = Matrix.subtract(layerActivations.get(structure.length-1), output);
        
        //from last layer to second layer 
        //(dont need to propagateBackward from layer 0)
        for(int layer = structure.length - 1; layer > 0; layer--){
            
            // initialize biasGradient same size as biases.get(layer)
            double[][] biasGradient = new double[biases.get(layer).length][1];
            // compute bias gradient
            for(int row = 0; row < biasGradient.length; row++){
                biasGradient[row][0] = layerGradient[row][0] *
                activationDerivative(layerInputs.get(layer)[row][0]);
            }
            // subtract from biasDeltas
            biasDeltas.set(layer, Matrix.subtract(biasDeltas.get(layer), biasGradient));
            
            
            
            // initialize weightGradient same size as weights.get(layer - 1)
            double[][] weightGradient = new double[weights.get(layer - 1).length][weights.get(layer - 1)[0].length];
            // compute weight gradient
            for(int toNeuron = 0; toNeuron < weightGradient.length; toNeuron++){
                for(int fromNeuron = 0; fromNeuron < weightGradient[0].length; fromNeuron++){
                    weightGradient[toNeuron][fromNeuron] = layerGradient[toNeuron][0] *
                    activationDerivative(layerInputs.get(layer)[toNeuron][0]) * 
                    layerActivations.get(layer - 1)[fromNeuron][0];
                }
            }
            // subtract from weightDeltas
            weightDeltas.set(layer - 1, Matrix.subtract(weightDeltas.get(layer - 1), weightGradient));
            
            
            
            // compute activation gradient (used as next layerGradient)
            double[][] activationGradient = new double[layerActivations.get(layer - 1).length][layerActivations.get(layer - 1)[0].length];
            for(int fromNeuron = 0; fromNeuron < activationGradient.length; fromNeuron++){
                for(int toNeuron = 0; toNeuron < structure[layer]; toNeuron++){
                    activationGradient[fromNeuron][0] += layerGradient[toNeuron][0] *
                    activationDerivative(layerInputs.get(layer)[toNeuron][0]) * 
                    weights.get(layer - 1)[toNeuron][fromNeuron];
                }
            }
            layerGradient = activationGradient;
        }
        
        /*double[][] layer3Difference = Matrix.subtract(layerActivations.get(2), output);
        
        for(int row=0; row<weightDeltas.get(1).length; row++){
            for(int col=0; col<weightDeltas.get(1)[0].length; col++){
                weightDeltas.get(1)[row][col] -= layer3Difference[row][0]*activationDerivative(layer3layerInputs[row][0])*layerActivations.get(1)[col][0]*learningRate;
            }
        }
        
        for(int row=0; row<weightDeltas.get(0).length; row++){
            for(int col=0; col<weightDeltas.get(0)[0].length; col++){
                weightDeltas.get(0)[row][col] -= layer3Difference[0][0]*
                activationDerivative(layer3layerInputs[0][0])*
                weights.get(1)[0][col]*  //check this
                activationDerivative(layer2layerInputs[col][0])*
                layerActivations.get(0)[row][0]*learningRate;
            }
        }
        
        
        for(int row=0; row<biases3Delta.length; row++){
            biases3Delta[row][0] -= layer3Difference[0][0]*
            activationDerivative(layer3layerInputs[row][0])*
            biasLearningRate;
        }
        
        for(int row=0; row<biases2Delta.length; row++){
            biases2Delta[row][0] -= layer3Difference[0][0]
            *activationDerivative(layer3layerInputs[0][0])
            *weights.get(1)[0][row]
            *activationDerivative(layer2layerInputs[row][0])
            *biasLearningRate;
        }
        
        add to running delta totals
        (these will be scaled and added to weights/biases in "update" functions)
        
        weightDeltas.get(0) +=
        weightDeltas.get(1) +=
        biases2Delta +=
        biases3Delta +=
        */
        
    }
    /**
     * @param trainingNum -- number of training examples (used to calc avg delta)
     */
    private void updateWeights(int trainingNum){
        for(int i=0; i<weights.size(); i++){
            weights.set(i, Matrix.add(weights.get(i), Matrix.multiply(learningRate/trainingNum, weightDeltas.get(i))));
            weightDeltas.set(i, new double[structure[i+1]][structure[i]]);
        }
    }
    /**
     * @param trainingNum -- number of training examples (used to calc avg delta)
     * @param normalized -- if true, the gradient is normalized (step size is constant.
     * If false, the step size is proportional to the magnitude of the gradient
     */
    private void updateWeights(int trainingNum, boolean normalized){
        if(normalized){
            for(int i=0; i<weights.size(); i++){
                weights.set(i, Matrix.add(weights.get(i), Matrix.multiply(learningRate, Matrix.normalize(weightDeltas.get(i)))));
                weightDeltas.set(i, new double[structure[i+1]][structure[i]]);
            }
        }else{
            updateWeights(trainingNum);
        }
    }
    /**
     * @param trainingNum -- number of training examples (used to calc avg delta)
     */
    private void updatebiases(int trainingNum){
        for(int i=1; i<biases.size(); i++){
            biases.set(i, Matrix.add(biases.get(i), Matrix.multiply(biasLearningRate/trainingNum, biasDeltas.get(i))));//add cumulative delta to bias
            biasDeltas.set(i, new double[structure[i]][1]);//reset delta counter
        }
    }
    /**
     * @param trainingNum -- number of training examples (only used if !normalized)
     * @param normalized -- if true, the gradient is normalized (step size is constant).
     * If false, the step size is proportional to the magnitude of the gradient
     */
    private void updatebiases(int trainingNum, boolean normalized){
        if(normalized){
            //biases.get(0) is null
            for(int i=1; i<biases.size(); i++){
                /* normalized gradient sum across training data batch,
                 * multiplies by learning rate, adds to each bias
                 */
                biases.set(i, Matrix.add(biases.get(i), Matrix.multiply(biasLearningRate, Matrix.normalize(biasDeltas.get(i)))));
                biasDeltas.set(i, new double[structure[i]][1]);
            }
        }else{
            updatebiases(trainingNum);
        }
    }
    /**
     * @param x -- value to be passes to activation function
     * @return -- activation function evaluated at x
     */
    private static double activationFunction(double x){
        return (1/(1+Math.pow(Math.E, -x)));
    }
    /**
     * @param x -- value to be passes to derviative activation function
     * @return -- derviative of activation function evaluated at x
     */
    private static double activationDerivative(double x){
        return (1/(Math.pow(Math.pow(Math.E, -x*.5)+Math.pow(Math.E, x*.5),2)));
    }
    /**
     * @param x -- value to be passes to error function
     * @return -- error function evaluated at x
     */
    private static double errorFunction(double expected, double actual){
        return 0.5 * Math.pow((expected - actual), 2);
    }
    /**
     * @param x -- value to be passes to derviative error function
     * @return -- derviative of error function evaluated at x
     */
    private static double errorDerivative(double expected, double actual){
        return actual - expected;
    }
    /**
     * @returns total error between arrays
     */
    private static double calcError(double[][] expected, double[][] actual){
        double error=0;
        if(expected.length != actual.length || expected[0].length != actual[0].length){
            throw new IllegalArgumentException("different dimensions for expected and actual data");
        }
        
        for(int row=0; row<expected.length; row++){
            for(int col=0; col<expected[0].length; col++){
                error += errorFunction(expected[row][col], actual[row][col]);
            }
        }
        return error;
    }
    /**
     * @param matrix -- double[][] to display to the console
     * @return -- string of matrix trimmed to maxCharsPerOutput length
     */
    private static String displayConsole(double[][] matrix){
        return Matrix.toStringOneLine(matrix, maxCharsPerOutput);
    }
    /**
     * @param error -- double to check if NaN
     * if error is NaN, the error has most likely diverged to infinity
     * and a RuntimeException is thrown
     */
    private static void checkError(double error){
        if(Double.isNaN(error)){
            throw new RuntimeException("error diverged to infinity. Try using a smaller learning rate");
        }
    }
    /**
     * @param data -- data to be converted from array of double[] to 
     * array of double[][] for program compatability
     * @return data as double[][][]
     */
    private static double[][][] convertData(double[][] data){
        double[][][] output = new double[data.length][data[0].length][1];
        
        for(int row = 0; row < data.length; row++){
           for(int col = 0; col < data[0].length; col++){
                output[row][col][0] = data[row][col];
            } 
        }
        
        return output;
    }
}

class Matrix{
    
    // matrix[row][col]
    
    public static double[][] transpose(double[][] a){
        double[][] output = new double[a.length][a[0].length];
        
        for(int row=0; row<output.length; row++){
            for(int col=0; col<output[0].length; col++){
                output[row][col] = a[col][row];
            }  
        }
        
        return output;
    }
    
    public static double[][] multiply(double[][] a, double[][] b){
        
        double[][] output = new double[a.length][b[0].length];
        
        for(int row=0; row < output.length; row++){
            for(int col=0; col < output[0].length; col++){
                output[row][col] = dotProduct(getRow(a, row), getCol(b, col));
            }
        }
        
        return output;
        
    }
    
    public static double[][] multiply(double scalar, double[][] b){
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < output.length; row++){
            for(int col=0; col < output[0].length; col++){
                output[row][col] = b[row][col]*scalar;
            }
        }
        
        return output;
        
    }
    
    public static double[][] multiply(int scalar, double[][] b){
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < output.length; row++){
            for(int col=0; col < output[0].length; col++){
                output[row][col] = b[row][col]*scalar;
            }
        }
        
        return output;
        
    }
    
    public static double[][] multiply(double[][] b, double scalar){
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < output.length; row++){
            for(int col=0; col < output[0].length; col++){
                output[row][col] = b[row][col]*scalar;
            }
        }
        
        return output;
        
    }
    
    public static double[][] multiply(double[][] b, int scalar){
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < output.length; row++){
            for(int col=0; col < output[0].length; col++){
                output[row][col] = b[row][col]*scalar;
            }
        }
        
        return output;
        
    }
    
    /**
     * @param index of row to return
     * @return row of matrix as an array
     */
    public static double[] getRow(double[][] a, int row){
        if(row>=a.length || row<0){
            throw new IllegalArgumentException("invalid row: " + row);
        }
        
        return a[row];
    }
    
    /**
     * @param index of column to return
     * @return column of matrix as an array
     */
    public static double[] getCol(double[][] a, int col){
        if(col>=a[0].length || col<0){
            throw new IllegalArgumentException("invalid column: " + col);
        }
        
        double[] output = new double[a.length];
        
        for(int i=0; i<output.length; i++){
            output[i] = a[i][col];
        }
        
        return output;
        
    }
    
    /**
     * @param a -- first 1d vector
     * @param b -- second 1d vector
     * @return dot product of two vectors
     */ 
    public static double dotProduct(double[] a, double[] b){
        if(a.length != b.length){
            throw new IllegalArgumentException("lengths of arguments unequal:" + a.length + " != " + b.length);
        }
        
        double sum = 0;
        
        for(int i=0; i < a.length; i++){
            sum+=a[i]*b[i];
        }
        
        return sum;
    }
    
    public static double[][] add(double[][] a, double[][] b){
        if(a.length != b.length && a[0].length != b[0].length){
            throw new IllegalArgumentException("matricies have different dimensions: " + a.length + " x " + a[0].length + " and " + + b.length + " x " + b[0].length);
        }
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < b.length; row++){
            for(int col=0; col < b[0].length; col++){
                output[row][col] = a[row][col] + b[row][col];
            }
        }
        
        return output;
        
    }
    
    public static double[][] subtract(double[][] a, double[][] b){
        if(a.length != b.length && a[0].length != b[0].length){
            throw new IllegalArgumentException("matricies have different dimensions: " + a.length + " x " + a[0].length + " and " + + b.length + " x " + b[0].length);
        }
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < b.length; row++){
            for(int col=0; col < b[0].length; col++){
                output[row][col] = a[row][col] - b[row][col];
            }
        }
        return output;
    }
    
    public static double[][] multiplyElementwise(double[][] a, double[][] b){
        if(a.length != b.length && a[0].length != b[0].length){
            throw new IllegalArgumentException("matricies have different dimensions: " + a.length + " x " + a[0].length + " and " + + b.length + " x " + b[0].length);
        }
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < b.length; row++){
            for(int col=0; col < b[0].length; col++){
                output[row][col] = a[row][col] * b[row][col];
            }
        }
        return output;
    }
    
    public static double[][] divideElementwise(double[][] a, double[][] b){
        if(a.length != b.length && a[0].length != b[0].length){
            throw new IllegalArgumentException("matricies have different dimensions: " + a.length + " x " + a[0].length + " and " + + b.length + " x " + b[0].length);
        }
        
        double[][] output = new double[b.length][b[0].length];
        
        for(int row=0; row < b.length; row++){
            for(int col=0; col < b[0].length; col++){
                output[row][col] = a[row][col] / b[row][col];
            }
        }
        return output;
    }
    
    public static String toString(double[][] a){
        String output = "{";
        
        for(int row=0; row<a.length; row++){
            output += "[";
            for(int col=0; col<a[0].length; col++){
                output += (a[row][col]);
                if(col!=a[0].length-1){
                    output += ", ";
                }
            }
            output += "]";
            if(row!=a.length-1){
                    output += ",\n";
            }
        }
        output += "}";
        return output;
    }
    public static String toString(double[][][] a){
        String output="[\n";
        for(int index=0; index<a.length; index++){
            output += (Matrix.toString(a[index])+"\n\n");
        }
        return output+"]";
    }
    
    public static String toString(double[] a){
        String output = "{";
        for(int col=0; col<a.length; col++){
            output += (a[col]);
            if(col!=a.length-1){
                output += ", ";
            }
        }
        output += "}";
        return output;
    }
    
    public static String toStringOneLine(double[][] a){
        if(a.length==1&&a[0].length==1){
            return a[0][0]+"";
        }
        
        String output = "{";
        
        for(int row=0; row<a.length; row++){
            output += "[";
            for(int col=0; col<a[0].length; col++){
                output += (a[row][col]);
                if(col!=a[0].length-1){
                    output += ", ";
                }
            }
            output += "]";
            if(row!=a.length-1){
                    output += ",";
            }
        }
        output += "}";
        return output;
    }
    public static String toStringOneLine(double[][] a, int maxChars){
        if(a.length==1&&a[0].length==1){
            return a[0][0]+"";
        }
        
        String output = "{";
        
        for(int row=0; row<a.length; row++){
            output += "[";
            for(int col=0; col<a[0].length; col++){
                output += (a[row][col]);
                if(col!=a[0].length-1){
                    output += ", ";
                }
            }
            output += "]";
            if(output.length() >= maxChars){
                return (output.substring(0, maxChars-3)+"...");
            }
            if(row!=a.length-1){
                    output += ",";
            }
        }
        
        output += "}";
        return output;
    }
    
    public static double[][] normalize(double[][]a){
        double sum=0;
        
        for(int row=0; row<a.length; row++){
            for(int col=0; col<a[0].length; col++){
                sum+=Math.pow(a[row][col], 2);
            }
        }
        
        if(sum == 0){
            return a;
        }
        
        sum=Math.sqrt(sum);
        
        for(int row=0; row<a.length; row++){
            for(int col=0; col<a[0].length; col++){
                a[row][col]/=sum;
            }
        }
        
        return a;
    }
    
    
    /**
     * 
     * replaces all values of given array with random doubles
     */
    public static double[][] randomizeArray(double[][] a){
        Random rand = new Random();
        for(int row=0; row<a.length; row++){
            for(int col=0; col<a[0].length; col++){
                a[row][col] = rand.nextDouble();
            }
        }
        
        return a;
    }
    
    /**
     * 
     * replaces all values of given array with random doubles using specified Random object
     */
    public static double[][] randomizeArray(double[][] a, Random rand){
        for(int row=0; row<a.length; row++){
            for(int col=0; col<a[0].length; col++){
                a[row][col] = rand.nextDouble();
            }
        }
        
        return a;
    }
}
