package spinach.argumentclassifier.featuregen;

import edu.stanford.nlp.util.ErasureUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class ExtensibleOnlineFeatureGenerator extends ArgumentFeatureGenerator {

    private Set<Integer> featureTypes;
    private List<IndividualFeatureGenerator> featureGeneratorList =
            new ArrayList<IndividualFeatureGenerator>();

    public ExtensibleOnlineFeatureGenerator() {
        this(new HashSet<Integer>());
    }

    public ExtensibleOnlineFeatureGenerator(Set<Integer> featureNums) {
        featureTypes = featureNums;
        addDefaultFeatures();
    }

    @Override
    public Collection<String> featuresOf(SemanticFrameSet sentenceAndPredicates,
                                         Token argument, Token predicate) {

        Collection<String> features = super.featuresOf(sentenceAndPredicates, argument, predicate);
        List<Token> featureTokens = new ArrayList<Token>();
        /* featureToken tokens:
        0: argument
        1: predicate
        2: ppHead
         */

        featureTokens.add(argument);
        featureTokens.add(predicate);

        Token ppHead;
        if (argument.syntacticHeadRelation.equals("PMOD"))
            ppHead = sentenceAndPredicates.getLeftSiblings(argument).getLast();
        else
            ppHead = sentenceAndPredicates.getParent(argument);
        featureTokens.add(ppHead);


        for (Integer i : featureTypes) {
            Collection<String> newFeatures = featureGeneratorList.get(i).
                    featuresOf(sentenceAndPredicates, featureTokens);
            if (newFeatures != null)
                features.addAll(newFeatures);
        }

        return features;

    }

    public boolean addFeatureType(int featureNum) {
        return featureTypes.add(featureNum);
    }

    public boolean removeFeatureType(int featureNum) {
        return featureTypes.remove(featureNum);
    }

    public int numFeatureTypes() {
        return featureGeneratorList.size();
    }

    public int numAddlFeatures() {
        return featureTypes.size();
    }

    public void clearFeatures() {
        featureTypes.clear();
    }

    public abstract class IndividualFeatureGenerator implements Serializable {
        abstract Collection<String> featuresOf(SemanticFrameSet frameSet, List<Token> featureTokens);
    }

    public int addFeature(IndividualFeatureGenerator featureGenerator) {
        featureGeneratorList.add(featureGenerator);
        return featureGeneratorList.size() - 1;
    }

    private void addDefaultFeatures() {
        addFeature(new IndividualFeatureGenerator() {
            //0: existSemDeprel
            @Override
            public Collection<String> featuresOf(SemanticFrameSet frameSet, List<Token> featureTokens) {
                //TODO
                return null;
            }
        });

        addFeature(new IndividualFeatureGenerator() {
            //1: previousArgClass
            @Override
            public Collection<String> featuresOf(SemanticFrameSet frameSet, List<Token> featureTokens) {
                List<String> feature = new ArrayList<String>(1);
                int mostRecentArgumentIndex = -1;
                String mostRecentLabel = null;

                Token predicate = featureTokens.get(1);
                for (Map.Entry<Token, String> pair : frameSet.argumentsOf(predicate).entrySet()) {
                    int currIndex = pair.getKey().sentenceIndex;
                    if (currIndex > mostRecentArgumentIndex) {
                        mostRecentArgumentIndex = currIndex;
                        mostRecentLabel = pair.getValue();
                    }
                }

                if (mostRecentLabel == null)
                    feature.add("previousArgClass:" + ArgumentClassifier.NIL_LABEL);
                else
                    feature.add("previousArgClass:" + mostRecentLabel);

                return feature;
            }
        });
    }

    private void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
        featureGeneratorList = ErasureUtils.uncheckedCast(in.readObject());
        featureTypes = ErasureUtils.uncheckedCast(in.readObject());

    }

    public static ExtensibleOnlineFeatureGenerator load(String path) throws ClassNotFoundException, IOException {
        GZIPInputStream is = new GZIPInputStream(new FileInputStream(path));

        ObjectInputStream in = new ObjectInputStream(is);
        ExtensibleOnlineFeatureGenerator ex = new ExtensibleOnlineFeatureGenerator();
        ex.load(in);
        in.close();
        is.close();
        return ex;
    }

    public void save(String path) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(path))));

            out.writeObject(featureGeneratorList);
            out.writeObject(featureTypes);

            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

