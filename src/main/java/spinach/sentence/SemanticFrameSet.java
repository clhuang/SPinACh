package spinach.sentence;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import edu.stanford.nlp.util.Pair;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * @author  Calvin Huang
 *
 * A SemanticFrameSet represents relationships between tokens
 * in a single sentence. Each SemanticFrameSet contains a sentence,
 * containing tokens and their syntactic relationships, and it also
 * contains a list of predicates, and mappings from those predicates
 * to their arguments, and the semantic relations.
 */
public class SemanticFrameSet{

    private TokenSentence sentence;
    private List<Token> predicateList = new ArrayList<Token>();
    private Multimap<Token, Pair<Token, String>> relations =
            HashMultimap.create();

    /**
     * Initialize the frameset with a sentence, and no predicates
     * or semantic relations.
     *
     * @param sentence sentence that the frameset will contain
     */
    public SemanticFrameSet(TokenSentence sentence){
        this.sentence = sentence;
    }

    /**
     * Get the contained sentence.
     *
     * @return sentence
     */
    public TokenSentence sentence(){
        return sentence;
    }

    /**
     * Add a predicate to the list of predicates.
     *
     * @param predicate predicate to be added
     */
    public void addPredicate(Token predicate){
        predicateList.add(predicate);
    }

    /**
     * Add a bunch of predicates to the list of predicates.
     * The predicates should be in order.
     *
     * @param predicateList list of predicates to be appended
     */
    public void addPredicates(List<Token> predicateList){
        predicateList.addAll(predicateList);
    }

    /**
     * Returns an immutable copy of the list of predicates.
     *
     * @return list of predicates
     */
    public List<Token> getPredicateList(){
        return Collections.unmodifiableList(predicateList);
    }

    /**
     * Adds a relation between an argument and a predicate.
     *
     * @param predicate predicate being referenced
     * @param argument argument being referenced
     * @param relation semantic relationship between the two
     */
    public void addArgument(Token predicate, Token argument, String relation){
        relations.put(predicate, new Pair<Token, String>(argument, relation));
    }

    /**
     * View the relations and arguments of some predicate
     *
     * @param predicate predicate to look at the arguments of
     * @return arguments of this predicate, and their relations
     */
    public Collection<Pair<Token, String>> argumentsOf(Token predicate){
        return relations.get(predicate);
    }

}
