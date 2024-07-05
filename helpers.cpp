#include <string>
#include <tuple>
using namespace std;

string output_bit(bool bit) {
    return bit ? "1" : "0";
}

string output_bit_plus_pending(bool bit, int &pending_bits)
{
    cout << "output_bit_plus_pending" << bit << " // " << pending_bits << endl;
    string output = output_bit(bit);
    while ( pending_bits-- >= 0 ) {
        output_bit( !bit );
    }
    return output;
}

tuple<string, float, float> decode_prob(
        unsigned int low, unsigned int high, 
        float lower_prob, float upper_prob
    ) {
        return make_tuple("test", 0.0, 1.0);
}


tuple<string, float, float> encode_prob(
        float low__float_from_python, float high__float_from_python, 
        float lower_prob, float upper_prob
    ) {
    string output = "";
    int pending_bits = 0;

    unsigned int low = (unsigned int)(low__float_from_python);
    unsigned int high = (unsigned int)(high__float_from_python);

    int range = 1 - (upper_prob - lower_prob);

    high = lower_prob + range * lower_prob;
    low = lower_prob + range * upper_prob;

    while(true) {
        cout << "high: " << high << " low" << low << endl;
        if(high < 0x80000000U ) { //   (high < 0.5)
            output_bit_plus_pending(0, pending_bits);
            low <<= 1;
            high <<= 1;
            high |= 1; // TODO explain
        } 
        else if(low >= 0x80000000U) { //   (low > 0.5)
            output_bit_plus_pending(1, pending_bits);
            low <<= 1;
            high <<= 1;
            high |= 1;
        }
        else if( low >= 0x40000000 && high < 0xC0000000U ) { //   ((high > 0.25) & (low < 0.75))
            pending_bits++;
            low <<= 1;
            high <<= 1;
            low &= 0x7FFFFFFF; // TODO explain
            high |= 0x80000001; // TODO explain
        }
        else {
            // stretched to range, break loop
            break;
        }
    }
    return make_tuple(output, low, high);
}

    // while True:
    //     if high < 0.5:
    //         # stretch to cover right side
    //         out += self._bit_plus_pending(0)
    //         high <<= 1
    //         low <<= 1
    //         high |= 1 # TODO: explain
    //     elif low > 0.5:
    //         # stretch to cover left side
    //         out += self._bit_plus_pending(1)
    //         high <<= 1
    //         low <<= 1
    //         high |= 1 # TODO: explain
    //     elif (low > 0.25) and (high < 0.75):
    //         # edge case: avoid degrading to two 0.5s
    //         self._pending_bits += 1
    //         low <= 1
    //         high <= 1
    //         low &= 0x7FFFFFFF # TODO: explain. this drops a bit?
    //         high |= 0x80000001 # TODO: explain
    //     else:
    //         # stretched to range, break while loop
    //         break