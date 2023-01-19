from azureml.core.compute import AmlCompute,ComputeTarget, ComputeInstance
from azureml.exceptions import ComputeTargetException
from azureml.core import Workspace
import argparse
import numpy as np

def create_cluster(
    workspaceRef,
    name:str,
    vmSize:str,
    minNodes:int,
    maxNodes:int,
    idleTime:int
    ):
    """Function to create compute cluster"""

    print(workspaceRef, name, vmSize,minNodes, maxNodes,idleTime)

    try:
        compute_target = AmlCompute(workspaceRef, name)
        print("Compute target already exists. Using it.")
        return compute_target
    except ComputeTargetException:
        print("Creating compute target")
        try:
            prov_config = AmlCompute.provisioning_configuration(vm_size=vmSize, min_nodes=minNodes,max_nodes=maxNodes, idle_seconds_before_scaledown=idleTime)
            compute_target = ComputeTarget.create(workspaceRef, name, prov_config)
            compute_target.wait_for_completion(show_output=True)
            return compute_target
        except:
            raise ComputeTargetException("Couldn't create cluster. Please enter valid inputs. If error persists, contact the developer.")
        
def create_instance(
    workspaceRef,
    name:str,
    vmSize:str
    ):
    """Function to create compute instance"""
    try:
        compute_target = ComputeInstance(workspaceRef, name)
        print("Compute target already exists. Using it.")
        return compute_target
    except ComputeTargetException:
        print("Creating compute target.")
        try:
            prov_config = ComputeInstance.provisioning_configuration(vm_size=vmSize)
            compute_target = ComputeTarget.create(workspaceRef, name, prov_config)
            compute_target.wait_for_completion(show_output=True)
            return compute_target
        except:
            raise ComputeTargetException("Couldn't create instance. Please enter valid inputs. If error persists, contact the developer.")

def delete_compute(
    workspaceRef,
    computeToDelete):
    """Function to delete compute cluster/instance"""
    try:
        compute = AmlCompute(workspaceRef, computeToDelete)
        compute.delete(wait_for_completion=True,show_output=True)
            
    except:
        try:
            compute = ComputeInstance(workspaceRef, computeToDelete)
            compute.delete(wait_for_completion=True,show_output=True)
        except:    
            print("Compute target does not exist.")

def view_compute(workspaceRef):

    try:
        computes=list(workspaceRef.compute_targets.keys())
        if len(computes)>0:
            print("List of compute targets: ", computes)
        else:
            print("No compute targets found in the workspace.")
    except:
        print("Please check the workspace reference.")

if __name__=="__main__":

    computeIdentifier = np.random.choice(10123)
    parser=argparse.ArgumentParser()
    parser.add_argument("--action", help="Options: 'create','delete','deleteAll','view'")
    parser.add_argument("--compute", help="Type of compute: 'cluster' or 'instance'", default="cluster", type=str)
    parser.add_argument("--name", default=f"compute{computeIdentifier}", type=str, help="Name of compute to create/delete.")
    parser.add_argument("--vmSize", help="Compute size", type=str)
    parser.add_argument("--shutDownTimeInSecs", default=180, type=int)
    parser.add_argument("--minNodes", default=0, help="Min. no. of nodes to create cluster.", type=int)
    parser.add_argument("--maxNodes", default=4, help="Max. no. of nodes to create cluster.", type=int)
    args=parser.parse_args()

    workspaceRef = Workspace.from_config(path='./config/config.json')

    if args.vmSize is None:
        if args.compute=="instance":
            args.vmSize="Standard_DS11_v2"
        else:
            args.vmSize="Standard_DS3_v2"
    if args.action=="create":
        if args.compute=="cluster":
            compute=create_cluster(
                workspaceRef,
                name=args.name,
                minNodes=args.minNodes,
                maxNodes=args.maxNodes,
                vmSize=args.vmSize,
                idleTime=args.shutDownTimeInSecs
            )
        elif args.compute=="instance":
            compute=create_instance(
                workspaceRef,
                name=args.name,
                vmSize=args.vmSize
            )
    elif args.action == "delete":
        try:
            delete_compute(workspaceRef, args.name)
        except:
            print("Could not delete compute.")

    elif args.action=="deleteAll":
        try:
            for compute_name in list(workspaceRef.compute_targets.keys()):
                delete_compute(workspaceRef, compute_name)
        except:
            print("Could not delete all computes.")
    elif args.action=="view":
        view_compute(workspaceRef)
    else:
        raise ValueError("Please enter a valid action between 'create' and 'delete'")
