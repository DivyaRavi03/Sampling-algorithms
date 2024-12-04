import itertools
import random
import math 
import argparse


class BayesNet:
    def __init__(self):
        # CPT based on Figure 13.2
        self.burglary_prob = 0.001  # P(B)
        self.earthquake_prob = 0.002  # P(E)
        
        # CPT for Alarm
        self.alarm_cpt = {
            (True, True): 0.7,    # P(A|B,E)
            (True, False): 0.01,   # P(A|B,¬E)
            (False, True): 0.7,   # P(A|¬B,E)
            (False, False): 0.01  # P(A|¬B,¬E)
        }
        
        # CPT for John and Mary calls
        self.john_call_cpt = {
            True: 0.90,   # P(J|A)
            False: 0.05   # P(J|¬A)
        }
        
        self.mary_call_cpt = {
            True: 0.70,   # P(M|A)
            False: 0.01   # P(M|¬A)
        }

    def exact_inference_enumeration(self, evidence, query):
        
    # Compute joint probability 
        def joint_probability(assignment):
            b, e, a, j, m = assignment['B'], assignment['E'], assignment['A'], assignment['J'], assignment['M']
            return (
                (self.burglary_prob if b else 1 - self.burglary_prob) *
                (self.earthquake_prob if e else 1 - self.earthquake_prob) *
                (self.alarm_cpt[(b, e)] if a else 1 - self.alarm_cpt[(b, e)]) *
                (self.john_call_cpt[a] if j else 1 - self.john_call_cpt[a]) *
                (self.mary_call_cpt[a] if m else 1 - self.mary_call_cpt[a])
                )


        variables = ['B', 'E', 'A', 'J', 'M']
        all_assignments = list(itertools.product([True, False], repeat=len(variables)))

    
        consistent_probs = {}
        for assignment in all_assignments:
            assignment_dict = dict(zip(variables, assignment))

        
            if all(assignment_dict[var] == value for var, value in evidence.items()):
           
                query_values = tuple(assignment_dict[var] for var in query)
                joint_prob = joint_probability(assignment_dict)

                if query_values not in consistent_probs:
                    consistent_probs[query_values] = 0
                consistent_probs[query_values] += joint_prob

    
        total_prob = sum(consistent_probs.values())
        for key in consistent_probs:
            consistent_probs[key] /= total_prob

        return consistent_probs


    def probability(self, variable, value, evidence):
        
        if variable == 'B':
            return self.burglary_prob if value else 1 - self.burglary_prob
        
        if variable == 'E':
            return self.earthquake_prob if value else 1 - self.earthquake_prob
        
        if variable == 'A':
            b = evidence.get('B', None)
            e = evidence.get('E', None)
            
            if b is None or e is None:
                return 0.5
            
            return self.alarm_cpt.get((b, e)) if value else 1 - self.alarm_cpt.get((b, e))
        
        if variable == 'J':
            a = evidence.get('A', None)
            if a is None:
                return 0.5
            
            return self.john_call_cpt.get(a) if value else 1 - self.john_call_cpt.get(a)
        
        if variable == 'M':
            a = evidence.get('A', None)
            if a is None:
                return 0.5
            
            return self.mary_call_cpt.get(a) if value else 1 - self.mary_call_cpt.get(a)

    def prior_sampling(self, evidence, query, num_samples):
        def generate_sample():
            
            sample = {}
            sample['B'] = random.random() < self.burglary_prob
            sample['E'] = random.random() < self.earthquake_prob
            
            
            alarm_prob = self.alarm_cpt.get((sample['B'], sample['E']))
            sample['A'] = random.random() < alarm_prob
            
            
            sample['J'] = random.random() < self.john_call_cpt.get(sample['A'])
            
            
            sample['M'] = random.random() < self.mary_call_cpt.get(sample['A'])
            
            return sample

        
        sample_counts = {q: {True: 0, False: 0} for q in query}
        total_valid_samples = 0

        for _ in range(num_samples):
            sample = generate_sample()
            
            
            evidence_match = all(sample.get(k) == v for k, v in evidence.items())
            
            if evidence_match:
                total_valid_samples += 1
                for q in query:
                    sample_counts[q][sample[q]] += 1

        
        query_probs = {}
        for q in query:
            if total_valid_samples == 0:
                query_probs[q] = 0.5  
            else:
                query_probs[q] = sample_counts[q][True] / total_valid_samples
        
        return query_probs

    def rejection_sampling(self, evidence, query, num_samples):
        return self.prior_sampling(evidence, query, num_samples)

    def likelihood_weighting(self, evidence, query, num_samples):
        def weighted_sample():
            sample = {}
            weight = 1.0

            
            sample['B'] = random.random() < self.burglary_prob
            sample['E'] = random.random() < self.earthquake_prob
            
            
            alarm_prob = self.alarm_cpt.get((sample['B'], sample['E']))
            sample['A'] = random.random() < alarm_prob
            
            
            sample['J'] = random.random() < self.john_call_cpt.get(sample['A'])
            sample['M'] = random.random() < self.mary_call_cpt.get(sample['A'])
            
            
            for var, val in evidence.items():
                if var in sample:
                    if sample[var] != val:
                        return sample, 0.0
                    weight *= self.probability(var, val, sample)
            
            return sample, weight

        
        sample_counts = {q: {True: 0.0, False: 0.0} for q in query}
        total_weight = 0.0

        for _ in range(num_samples):
            sample, weight = weighted_sample()
            
            if weight > 0:
                total_weight += weight
                for q in query:
                    sample_counts[q][sample[q]] += weight

        
        query_probs = {}
        for q in query:
            if total_weight == 0:
                query_probs[q] = 0.5
            else:
                query_probs[q] = sample_counts[q][True] / total_weight
        
        return query_probs

def run_inference_comparison(net, evidence, query, sample_sizes, runs):
    results = {
        'Prior Sampling': {},
        'Rejection Sampling': {},
        'Likelihood Weighting': {},
        'Exact': {}
    }

    
    exact_result = net.exact_inference_enumeration(evidence, query)
    results['Exact'] = exact_result

    
    for num_samples in sample_sizes:
        # Average over 10 runs for each sampling method
        prior_runs = [net.prior_sampling(evidence, query, num_samples) for _ in range(runs)]
        rejection_runs = [net.rejection_sampling(evidence, query, num_samples) for _ in range(runs)]
        lw_runs = [net.likelihood_weighting(evidence, query, num_samples) for _ in range(runs)]

        # Compute averages
        results['Prior Sampling'][num_samples] = {
            q: sum(run[q] for run in prior_runs) / 10 for q in query
        }
        results['Rejection Sampling'][num_samples] = {
            q: sum(run[q] for run in rejection_runs) / 10 for q in query
        }
        results['Likelihood Weighting'][num_samples] = {
            q: sum(run[q] for run in lw_runs) / 10 for q in query
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Network Inference")
    parser.add_argument("--sample-size", type=int, required=True,
                        help="Number of samples to use for sampling algorithms (required)")
    parser.add_argument("--runs", type=int, required=True,
                        help="Number of samples to use for sampling algorithms (required)")
    args = parser.parse_args()
    net = BayesNet()
    #sample_sizes = [10, 50, 100, 200, 500, 1000, 10000]
    sample_sizes = [args.sample_size]
    runs = args.runs
    # Query 1
    print("Query 1: Alarm is false, infer Burglary and JohnCalls being true")
    evidence1 = {'A': False}
    query1 = ['B', 'J']
    results1 = run_inference_comparison(net, evidence1, query1, sample_sizes,runs)
    print_results(results1, query1)

    # Query 2
    print("\nQuery 2: JohnCalls is true, Earthquake is false, infer Burglary and MaryCalls being true")
    evidence2 = {'J': True, 'E': False}
    query2 = ['B', 'M']
    results2 = run_inference_comparison(net, evidence2, query2, sample_sizes,runs)
    print_results(results2, query2)

    # Query 3
    print("\nQuery 3: MaryCalls is true and JohnCalls is false, infer Burglary and Earthquake being true")
    evidence3 = {'M': True, 'J': False}
    query3 = ['B', 'E']
    results3 = run_inference_comparison(net, evidence3, query3, sample_sizes,runs)
    print_results(results3, query3)

def print_results(results, query):
    
    exact_results = [
        f"<{', '.join([str(q) for q in query])}, {results['Exact'][key]:.10f}>"
        for key in results['Exact']
    ]
    print(f"Exact Inference: {exact_results[0]}")

    # Print sampling results for each method and sample size
    for method in ['Prior Sampling', 'Rejection Sampling', 'Likelihood Weighting']:
        print(f"Results for sampling method: {method}")
        for num_samples in sorted(results[method].keys()):  
            probs = [
                f"<{', '.join([str(q) for q in query])}, {results[method][num_samples][q]:.10f}>"
                for q in query
            ]
            print(f"Number of samples: {num_samples}, Results: {', '.join(probs)}")


if __name__ == "__main__":
    main()