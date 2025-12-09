# evaluation_loop.py
"""
Evaluation loop for AITW dataset with XMLAgent and VLMAgent.
"""

import tensorflow as tf
from collections import defaultdict
from PIL import Image
import io
import numpy as np
from agents.vlm_agent import VLMAgent
from agents.llm_agent import XMLAgent
from typing import Dict, Any, List
from actions.action_parser import parse_action
from actions.action_matching import match_action


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode image bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def parse_tfrecord_example(example_bytes):
    """Parse a single TFRecord example."""
    feature_description = {
        'screenshot': tf.io.FixedLenFeature([], tf.string),
        'xml_tree': tf.io.FixedLenFeature([], tf.string),
        'episode_id': tf.io.FixedLenFeature([], tf.string),
        'step_id': tf.io.FixedLenFeature([], tf.int64),
        'action_type': tf.io.FixedLenFeature([], tf.int64),
        'action_x': tf.io.FixedLenFeature([], tf.float32),
        'action_y': tf.io.FixedLenFeature([], tf.float32),
    }
    
    parsed = tf.io.parse_single_example(example_bytes, feature_description)
    return {
        'screenshot': parsed['screenshot'].numpy(),
        'xml_tree': parsed['xml_tree'].numpy().decode('utf-8'),
        'episode_id': parsed['episode_id'].numpy().decode('utf-8'),
        'step_id': int(parsed['step_id'].numpy()),
        'action_type': int(parsed['action_type'].numpy()),
        'action_x': float(parsed['action_x'].numpy()),
        'action_y': float(parsed['action_y'].numpy()),
    }


def group_episodes(dataset) -> Dict[str, List[Dict[str, Any]]]:
    """Group dataset items by episode_id."""
    episodes = defaultdict(list)
    
    for example_bytes in dataset:
        try:
            parsed = parse_tfrecord_example(example_bytes)
            episodes[parsed['episode_id']].append(parsed)
        except Exception as e:
            print(f"Error parsing example: {e}")
            continue
    
    # Sort steps within each episode by step_id
    for episode_id in episodes:
        episodes[episode_id].sort(key=lambda x: x['step_id'])
    
    return dict(episodes)


def evaluate_agents(dataset, xml_agent: XMLAgent, vlm_agent: VLMAgent, 
                   parse_action, match_action):
    """
    Main evaluation loop for both agents.
    
    Args:
        dataset: TensorFlow TFRecordDataset
        xml_agent: XMLAgent instance
        vlm_agent: VLMAgent instance
        parse_action: Function to parse model output
        match_action: Function to match predicted and ground truth actions
    """
    
    print("Grouping episodes...")
    episodes = group_episodes(dataset)
    print(f"Found {len(episodes)} episodes")
    
    # Initialize metrics
    xml_metrics = {
        'action_correct': 0,
        'full_correct': 0,
        'total': 0
    }
    
    vlm_metrics = {
        'action_correct': 0,
        'full_correct': 0,
        'total': 0
    }
    
    episode_results = []
    
    # Evaluate each episode
    for episode_idx, (episode_id, steps) in enumerate(episodes.items()):
        print(f"\nEvaluating episode {episode_idx + 1}/{len(episodes)}: {episode_id}")
        
        episode_xml_correct = 0
        episode_xml_full = 0
        episode_vlm_correct = 0
        episode_vlm_full = 0
        episode_total = 0
        
        for step in steps:
            try:
                # Decode screenshot
                image = decode_image(step['screenshot'])
                
                # Construct state
                state = {
                    'task': 'Complete the task',  # Placeholder
                    'xml': step['xml_tree'],
                    'image': image,
                    'episode_id': step['episode_id'],
                    'step_id': step['step_id']
                }
                
                # Ground truth
                ground_truth = {
                    'action_type': step['action_type'],
                    'x': step['action_x'],
                    'y': step['action_y']
                }
                
                # Evaluate XMLAgent
                try:
                    xml_output = xml_agent.run_step(state)
                    xml_parsed = parse_action(xml_output)
                    xml_result = match_action(xml_parsed, ground_truth)
                    
                    if xml_result['action_acc']:
                        xml_metrics['action_correct'] += 1
                        episode_xml_correct += 1
                    if xml_result['full_acc']:
                        xml_metrics['full_correct'] += 1
                        episode_xml_full += 1
                    xml_metrics['total'] += 1
                    episode_total += 1
                    
                except Exception as e:
                    print(f"XMLAgent error on episode {episode_id}, step {step['step_id']}: {e}")
                
                # Evaluate VLMAgent
                try:
                    vlm_output = vlm_agent.run_step(state)
                    vlm_parsed = parse_action(vlm_output)
                    vlm_result = match_action(vlm_parsed, ground_truth)
                    
                    if vlm_result['action_acc']:
                        vlm_metrics['action_correct'] += 1
                        episode_vlm_correct += 1
                    if vlm_result['full_acc']:
                        vlm_metrics['full_correct'] += 1
                        episode_vlm_full += 1
                    vlm_metrics['total'] += 1
                    
                except Exception as e:
                    print(f"VLMAgent error on episode {episode_id}, step {step['step_id']}: {e}")
                
            except Exception as e:
                print(f"Error processing step {step['step_id']} in episode {episode_id}: {e}")
                continue
        
        # Store episode results
        episode_results.append({
            'episode_id': episode_id,
            'total_steps': episode_total,
            'xml_action_acc': episode_xml_correct / episode_total if episode_total > 0 else 0,
            'xml_full_acc': episode_xml_full / episode_total if episode_total > 0 else 0,
            'vlm_action_acc': episode_vlm_correct / episode_total if episode_total > 0 else 0,
            'vlm_full_acc': episode_vlm_full / episode_total if episode_total > 0 else 0,
        })
    
    # Print overall results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nXMLAgent Results:")
    print(f"  Total steps evaluated: {xml_metrics['total']}")
    if xml_metrics['total'] > 0:
        print(f"  Action accuracy: {xml_metrics['action_correct'] / xml_metrics['total']:.4f} "
              f"({xml_metrics['action_correct']}/{xml_metrics['total']})")
        print(f"  Full accuracy: {xml_metrics['full_correct'] / xml_metrics['total']:.4f} "
              f"({xml_metrics['full_correct']}/{xml_metrics['total']})")
    
    print(f"\nVLMAgent Results:")
    print(f"  Total steps evaluated: {vlm_metrics['total']}")
    if vlm_metrics['total'] > 0:
        print(f"  Action accuracy: {vlm_metrics['action_correct'] / vlm_metrics['total']:.4f} "
              f"({vlm_metrics['action_correct']}/{vlm_metrics['total']})")
        print(f"  Full accuracy: {vlm_metrics['full_correct'] / vlm_metrics['total']:.4f} "
              f"({vlm_metrics['full_correct']}/{vlm_metrics['total']})")
    
    print("\n" + "="*80)
    print("PER-EPISODE SUMMARY")
    print("="*80)
    
    for result in episode_results:
        print(f"\nEpisode: {result['episode_id']}")
        print(f"  Steps: {result['total_steps']}")
        print(f"  XMLAgent - Action: {result['xml_action_acc']:.4f}, Full: {result['xml_full_acc']:.4f}")
        print(f"  VLMAgent - Action: {result['vlm_action_acc']:.4f}, Full: {result['vlm_full_acc']:.4f}")
    
    return xml_metrics, vlm_metrics, episode_results


def main():
    """Main entry point for evaluation."""
    # Load dataset
    filenames = tf.io.gfile.glob("/data/rxhuang/aitw/*")
    dataset = tf.data.TFRecordDataset(filenames)
    
    # Initialize agents
    xml_agent = XMLAgent(model_name="claude-sonnet-4-20250514")
    vlm_agent = VLMAgent(model_name="claude-sonnet-4-20250514")
    
    # Run evaluation
    evaluate_agents(dataset, xml_agent, vlm_agent, parse_action, match_action)


if __name__ == "__main__":
    main()
    