package spinach.sentence;

import java.util.Collections;
import java.util.HashMap;
import java.util.ListIterator;
import java.util.Map;

/**
 * A SemanticFrameSet represents relationships between tokens
 * in a single sentence. Each SemanticFrameSet contains a sentence,
 * containing tokens and their syntactic relationships, and it also
 * contains a list of predicates, and mappings from those predicates
 * to their arguments, and the semantic relations.
 *
 * @author Calvin Huang
 */
public class SemanticFrameSet extends TokenSentenceAndPredicates {

    private final Map<Token, Map<Token, String>> relations = new HashMap<Token, Map<Token, String>>();

    /**
     * Creates an empty SemanticFrameSet.
     */
    public SemanticFrameSet() {
    }

    /**
     * Creates a SemanticFrameSet using predicates and tokens from a TokenSentenceAndPredicates.
     *
     * @param sentenceAndPredicates sentence with predicates and token data
     */
    public SemanticFrameSet(TokenSentenceAndPredicates sentenceAndPredicates) {
        super(sentenceAndPredicates);
        predicateList = sentenceAndPredicates.predicateList;
    }

    /**
     * Adds a relation between an argument and a predicate.
     *
     * @param predicate predicate being referenced
     * @param argument  argument being referenced
     * @param relation  semantic relationship between the two
     */
    public void addArgument(Token predicate, Token argument, String relation) {
        //relations.put(predicate, new Pair<Token, String>(argument, relation));
        Map<Token, String> predicateMap = relations.get(predicate);
        if (predicateMap == null) {
            predicateMap = new HashMap<Token, String>();
            predicateMap.put(argument, relation);
            relations.put(predicate, predicateMap);
        } else
            predicateMap.put(argument, relation);

    }

    /**
     * View the relations and arguments of some predicate
     *
     * @param predicate predicate to look at the arguments of
     * @return arguments of this predicate, and their relations
     */
    public Map<Token, String> argumentsOf(Token predicate) {
        Map<Token, String> m = relations.get(predicate);
        if (m == null)
            return new HashMap<Token, String>();
        return Collections.unmodifiableMap(m);
    }

    /**
     * Trim the list of predicates--any predicate without arguments is removed
     */
    public void trimPredicates() {
        for (ListIterator<Token> iterator = predicateList.listIterator(predicateList.size());
             iterator.hasPrevious(); ) {
            Map<Token, String> arguments = argumentsOf(iterator.previous());
            if (arguments == null || arguments.isEmpty())
                iterator.remove();
        }
    }

}
