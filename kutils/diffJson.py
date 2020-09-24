import sys
import json

def checkDifference(orig, new):
    diff = {}
    if type(orig) != type(new):
        # print "Type difference"
        return True
    else:
        if type(orig) is dict and type(new) is dict:
            # print "Types are both dicts"
            # Check each of these dicts from the key level
            diffTest = False
            keys = sorted(orig)
            for key in keys:
                result = checkDifference(orig[key], new[key])
                if result != False:  ## Means a difference was found and returned
                    diffTest = True
                    # print "key/Values different: " + str(key)
                    diff[key] = result
                    #print('Diff in key level')
            # And check for keys in second dataset that aren't in first
            news = sorted(new)
            for key in news:
                if key not in orig:
                    diff[key] = ("KeyNotFound", new[key])
                    diffTest = True
                    #print('Key not found')

            if diffTest:
                return diff
            else:
                return False
        elif type(orig) is list and type(new) is list:
            if len(orig) != len(new):
                return True
            diffTest = False
            for ind in range(len(orig)):
                result = checkDifference(orig[ind], new[ind])
                if result != False:  ## Means a difference was found and returned
                    diffTest = True
                    # print "key/Values different: " + str(key)
                    diff['#{}'.format(ind)] = result
                    print('Diff in key level')
            if diffTest:
                return diff
            else:
                return False
        else:
            # print "Types were not dicts, likely strings"
            if str(orig) == str(new):
                return False
            else:
                return (str(orig), str(new))
    return diff

if __name__ == "__main__":
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    fo = sys.argv[3]

    with open(f1) as old_file, open(f2) as new_file:
        obj1 = json.load(old_file)
        obj2 = json.load(new_file)
        diff_res = checkDifference(obj1, obj2)

        outs = json.dumps(diff_res, indent=4, ensure_ascii=False)
        outf = open(fo, "w")
        print(outs, file=outf)
        outf.close()
