package spinach;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * A class for various utilities that deal with text corpuses.
 *
 * @author Calvin Huang
 */
public class CorpusUtils {

    public static final int INDEX_COLUMN = 0;
    public static final int FORM_COLUMN = 5;
    public static final int LEMMA_COLUMN = 6;
    public static final int POS_COLUMN = 7;
    public static final int PARENT_INDEX_COLUMN = 8;
    public static final int SEMANTIC_RELATION_COLUMN = 9;
    public static final int PREDICATE_COLUMN = 10;
    public static final int ARGS_START_COLUMN = 11;

    /**
     * Given the location of a syntactically annotated text corpus, returns a list of semantic frames.
     *
     * @param corpusLoc text corpus file location
     * @return list of semantic framesets, one for each sentence in the corpus
     */
    public static List<SemanticFrameSet> parseCorpus(String corpusLoc) throws IOException {

        List<SemanticFrameSet> sentences =
                new ArrayList<SemanticFrameSet>();

        String strLine;
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(corpusLoc)));

        ArrayList<String[]> sentenceTokenData = new ArrayList<String[]>();

        while ((strLine = br.readLine()) != null) {
            if (strLine.equals("")) {    //sentence is over, enter sentence data into set

                SemanticFrameSet goldFrames = new SemanticFrameSet();

                for (String[] tokenData : sentenceTokenData) {   //add tokens into sentence; predicates into frames
                    Token token = new Token(
                            tokenData[FORM_COLUMN],
                            tokenData[LEMMA_COLUMN],
                            tokenData[POS_COLUMN],
                            tokenData[SEMANTIC_RELATION_COLUMN],
                            Integer.parseInt(tokenData[PARENT_INDEX_COLUMN]) - 1,
                            Integer.parseInt(tokenData[INDEX_COLUMN]) - 1
                    );      //subtract 1 from indices because corpus is 1-based, but code uses 0-base

                    goldFrames.addToken(token);

                    if (!tokenData[PREDICATE_COLUMN].equals("_"))   //is a predicate
                        goldFrames.addPredicate(token);
                }

                List<Token> predicates = goldFrames.getPredicateList();

                for (String[] tokenData : sentenceTokenData) {    //link arguments to predicates
                    Token thisToken = goldFrames.tokenAt(Integer.parseInt(tokenData[INDEX_COLUMN]) - 1);
                    for (int i = ARGS_START_COLUMN; i < tokenData.length; i++)
                        if (!tokenData[i].equals("_")) {
                            int predicateNum = i - ARGS_START_COLUMN;
                            goldFrames.addArgument(predicates.get(predicateNum),
                                    thisToken,
                                    tokenData[i]);
                        }
                }

                sentences.add(goldFrames);

                sentenceTokenData.clear();
            } else {   //is token in same sentence
                sentenceTokenData.add(strLine.split("\\s+"));
            }
        }

        return sentences;

    }

}
