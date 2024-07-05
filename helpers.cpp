#include <string>
#include <tuple>
using namespace std;

string output_bit(bool bit) {
    return bit ? "1" : "0";
}

string output_bit_plus_pending(bool bit, int &pending_bits)
{
    cout << "[output_bit_plus_pending]  " << bit << " // " << pending_bits << endl;
    string output = output_bit(bit);
    while ( pending_bits > 0 ) {
        cout << "\t[output_bit_plus_pending in-loop]  " << bit << " // " << pending_bits << endl;
        output += output_bit( !bit );
        pending_bits --;
    }
    return output;
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
            output += output_bit_plus_pending(0, pending_bits);
            low <<= 1;
            high <<= 1;
            high |= 1; // TODO explain
        } 
        else if(low >= 0x80000000U) { //   (low > 0.5)
            output += output_bit_plus_pending(1, pending_bits);
            pending_bits = 0;
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



tuple<string, float, float> decode_prob(
        string message, 
        float low__float_from_python, float high__float_from_python, 
        float value__float_from_python,
        float lower_prob, float upper_prob
    ) {
    string output = "";
    unsigned int low = (unsigned int)(low__float_from_python);
    unsigned int high = (unsigned int)(high__float_from_python);
    unsigned int value = (unsigned int)(value__float_from_python);

    // now read in char-by-char
    unsigned int range = high - low + 1;
    high = low + (range*upper_prob) - 1;
    low = low + (range*lower_prob);
    while(true) {
        cout << "[decode_prob.inner]  low -> " << low << " high ->" << high << " message.length ->" << message.length() << endl; 
        if ( low >= 0x80000000U || high < 0x80000000U ) {
            low <<= 1;
            high <<= 1;
            high |= 1;
            value <<= 1;
            char next_bit = message[0]; message = message.substr(1);
            value += (next_bit == '1') ? 1 : 0;
        } 
        else if ( low >= 0x40000000 && high < 0xC0000000U ) {
            low <<= 1;
            low &= 0x7FFFFFFF;
            high <<= 1;
            high |= 0x80000001;
            value -= 0x4000000;
            value <<= 1;
            char next_bit = message[0]; message = message.substr(1);
            value += (next_bit == '1') ? 1 : 0;
        } 
        else {
            cout << " break!";
            break;
        }
    }
    return make_tuple(message, low, high);
}