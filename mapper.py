import dspy
from dspy.evaluate.metrics import answer_exact_match
from typing import List


# Define a structured signature for field matching
class FieldMatch(dspy.Signature):
    """Match a source field to its semantic equivalent in target fields."""

    source_field: str = dspy.InputField(desc="The source field name to match")
    target_fields: List[str] = dspy.InputField(
        desc="List of target field names to match against"
    )

    source: str = dspy.OutputField(desc="The original source field")
    match: str = dspy.OutputField(desc="The best matching target field")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    explanation: str = dspy.OutputField(
        desc="Explanation of why they match semantically"
    )


# Define a structured signature for batch matching
class BatchFieldMatches(dspy.Signature):
    """Match multiple source fields to their semantic equivalents in target fields."""

    source_fields: List[str] = dspy.InputField(
        desc="List of source field names to match"
    )
    target_fields: List[str] = dspy.InputField(
        desc="List of target field names to match against"
    )
    matches: List[FieldMatch] = dspy.OutputField(
        desc="Mapping of each source field to a corresponding FieldMatch (containing source, match, confidence, explanation)"
    )


class ConfigUpdate(dspy.Signature):
    """Updates and prettifies the target configuration with matched fields."""

    target_config: dict = dspy.InputField(
        desc="The unmapped target configuration to update with matches"
    )
    matches: List[FieldMatch] = dspy.InputField(
        desc="The matches to insert into the target config. Each match's source should replace the empty selector"
    )
    updated_config: dict = dspy.OutputField(
        desc="The updated target configuration with selectors present"
    )


# Define a module that learns to match fields
class FieldMatchPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.match_field = dspy.Predict(FieldMatch)

    def match(self, source_field, target_fields):
        return self.match_field(source_field=source_field, target_fields=target_fields)


# Define a module that learns to batch match fields
class BatchFieldMatchPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.batch_match = dspy.Predict(BatchFieldMatches)

    def match(self, source_fields, target_fields):
        return self.batch_match(
            source_fields=source_fields, target_fields=target_fields
        )


class ConfigUpdater(dspy.Module):
    def __init__(self):
        super().__init__()
        self.updater = dspy.ChainOfThought(ConfigUpdate)

    def update(self, target_config, matches):
        return self.updater(target_config=target_config, matches=matches)


# Main mapper class that can be optimized
class AutoMapper:
    def __init__(self, model_name="openai/gpt-4o-mini"):
        """
        Initialize a DSPy-based field matcher.

        Args:
           model_name: The LM to use for DSPy
        """
        self.lm = dspy.LM(model=model_name)
        dspy.configure(lm=self.lm)

        # Create the predictors
        self.batch_matcher = BatchFieldMatchPredictor()
        self.updater = ConfigUpdater()

        # Set up optimized versions
        self.optimized_batch_matcher = None

    def batch_match_fields(self, source_fields, target_fields):
        """
        Match all fields at once using DSPy's structured prediction.

        Args:
              source_fields: List of source field names
              target_fields: List of target field names
              threshold: Minimum confidence for a valid match

        Returns:
              Dictionary mapping each source field to its match information
        """
        # Use optimized matcher if available, otherwise use the base matcher
        matcher = self.optimized_batch_matcher or self.batch_matcher
        result = matcher.match(source_fields, target_fields)
        return result.matches

    def update_config_with_matches(self, target_config, matches):
        result = self.updater.update(target_config, matches)
        return result.updated_config

    def optimize_with_examples(self, example_pairs):
        """
        Optimize the matchers using example pairs of fields.

        Args:
           example_pairs: List of (source_field, target_field, target_fields) tuples
        """
        # Create training examples for single field matching
        train_data = []
        for source, target, all_targets in example_pairs:
            # Create a training example
            example = dspy.Example(
                source_field=source,
                target_fields=all_targets,
                match=target,
                confidence=1.0,
                explanation=f"'{source}' semantically matches '{target}' because they represent the same concept in different notation systems.",
            ).with_inputs("source_field", "target_fields")

            train_data.append(example)

        # Optimize the single field matcher
        print("Optimizing single field matcher...")
        optimizer = dspy.BootstrapFewShot(metric=answer_exact_match)
        self.optimized_single_matcher = optimizer.compile(
            FieldMatchPredictor(), trainset=train_data[: len(train_data) // 2]
        )

        # Create training data for batch matching
        batch_train_data = []

        # Group examples by unique target_fields lists
        grouped_examples = {}
        for source, target, all_targets in example_pairs:
            target_tuple = tuple(all_targets)  # Convert list to tuple for hashability
            if target_tuple not in grouped_examples:
                grouped_examples[target_tuple] = []
            grouped_examples[target_tuple].append((source, target))

        # Create batch examples
        for target_fields, source_target_pairs in grouped_examples.items():
            source_fields = [pair[0] for pair in source_target_pairs]
            matches = {}
            for source, target in source_target_pairs:
                matches[source] = {
                    "match": target,
                    "confidence": 1.0,
                    "explanation": f"'{source}' semantically matches '{target}'",
                }

            # Create a batch training example
            if source_fields and target_fields:  # Ensure non-empty
                example = dspy.Example(
                    source_fields=source_fields,
                    target_fields=list(target_fields),  # Convert back to list
                    matches=matches,
                ).with_inputs("source_fields", "target_fields")

                batch_train_data.append(example)

        # Optimize the batch matcher if we have enough examples
        if len(batch_train_data) >= 2:
            print("Optimizing batch field matcher...")
            batch_optimizer = dspy.BootstrapFewShot(metric=self._batch_accuracy_metric)
            self.optimized_batch_matcher = batch_optimizer.compile(
                BatchFieldMatchPredictor(),
                trainset=batch_train_data[: len(batch_train_data) // 2],
            )

    def _accuracy_metric(self, example, prediction):
        """Metric function for single field matching optimization."""
        # Check if the predicted match matches the expected match
        return float(prediction.match == example.match)

    def _batch_accuracy_metric(self, example, prediction):
        """Metric function for batch field matching optimization."""
        correct = 0
        total = 0

        # Check each source field in the example
        for source_field, expected_match_info in example.matches.items():
            expected_match = expected_match_info["match"]

            # Get the predicted match for this source field
            predicted_match_info = prediction.matches.get(source_field, {})
            predicted_match = (
                predicted_match_info.get("match")
                if isinstance(predicted_match_info, dict)
                else None
            )

            # Count correct matches
            if predicted_match == expected_match:
                correct += 1
            total += 1

        # Return the accuracy for this batch example
        return float(correct) / float(max(1, total))
